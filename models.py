"""
Model definitions (transfer learning with timm ImageNet backbones) + checkpoint I/O.

Task A  DiseaseNet     : backbone -> dropout -> linear(8)  -> sigmoid (multi-label)
Task B  RefractiveNet  : backbone -> MLP head -> spherical equivalent (diopters)
                         optional extra heads: sphere, cylinder, axis (as cos2θ, sin2θ)
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import torch
import torch.nn as nn

import config


# ---------------------------------------------------------------------------
# Backbones (timm). ImageNet weights are downloaded automatically by timm, or
# read from models/pretrained/<arch>*.pth when that file exists (offline use).
# ---------------------------------------------------------------------------
PRETRAINED_DIR = config.MODELS_DIR / "pretrained"


def _local_weights(arch: str):
    import os
    d = Path(os.environ.get("EYEVISION_PRETRAINED_DIR", PRETRAINED_DIR))
    hits = sorted(d.glob(f"{arch}_*.pth")) + sorted(d.glob(f"{arch}.pth"))
    return hits[0] if hits else None


def build_backbone(arch: str, pretrained: bool = True):
    """Return (backbone returning pooled features, feature_dim)."""
    import timm
    if arch not in ("efficientnet_b0", "efficientnet_b2", "resnet50", "resnet18", "convnext_tiny"):
        raise ValueError(f"Unknown architecture '{arch}'")
    kw = dict(num_classes=0, drop_rate=0.0)
    if arch.startswith("efficientnet") or arch.startswith("convnext"):
        kw["drop_path_rate"] = 0.1
    local = _local_weights(arch) if pretrained else None
    if local is not None:
        m = timm.create_model(arch, pretrained=True, pretrained_cfg_overlay={"file": str(local)}, **kw)
    else:
        m = timm.create_model(arch, pretrained=pretrained, **kw)
    return m, m.num_features


def gradcam_layer(model: nn.Module) -> nn.Module:
    """Last convolutional block - the usual Grad-CAM target."""
    b = model.backbone
    if hasattr(b, "layer4"):          # ResNet
        return b.layer4
    if hasattr(b, "bn2"):             # EfficientNet: conv_head -> bn2(+act)
        return b.bn2
    return b.stages[-1]               # ConvNeXt


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def early_modules(model: nn.Module):
    """First ~half of the backbone, frozen in CPU-friendly partial fine-tuning (--freeze-early)."""
    b = model.backbone
    if hasattr(b, "layer4"):                                   # ResNet: stem + layer1-2
        return [b.conv1, b.bn1, b.layer1, b.layer2]
    if hasattr(b, "blocks"):                                   # EfficientNet: stem + first 3 stages
        return [b.conv_stem, b.bn1, *list(b.blocks)[:3]]
    return [b.stem, *list(b.stages)[:2]]                       # ConvNeXt


def freeze_early(model: nn.Module):
    for mod in early_modules(model):
        for p in mod.parameters():
            p.requires_grad = False


def early_bn_eval(model: nn.Module):
    """Keep BatchNorm statistics of frozen early layers fixed while training."""
    for mod in early_modules(model):
        for m in mod.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


# ---------------------------------------------------------------------------
# Task A - disease classification
# ---------------------------------------------------------------------------
class DiseaseNet(nn.Module):
    def __init__(self, arch="efficientnet_b0", num_classes=config.NUM_DISEASES, pretrained=True, dropout=0.3):
        super().__init__()
        self.arch = arch
        self.backbone, dim = build_backbone(arch, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward_head(self, feats):
        return self.head(self.dropout(feats))                  # logits

    def forward(self, x):
        return self.forward_head(self.backbone(x))


# ---------------------------------------------------------------------------
# Task B - refractive error regression
# ---------------------------------------------------------------------------
class RefractiveNet(nn.Module):
    """Outputs a (B, 5) tensor: [SE (standardised), sphere, cylinder, cos2θ, sin2θ].
    Only column 0 is trained unless multihead=True and the labels exist."""

    def __init__(self, arch="resnet50", pretrained=True, dropout=0.3, multihead=False):
        super().__init__()
        self.arch = arch
        self.multihead = multihead
        self.backbone, dim = build_backbone(arch, pretrained)
        self.shared = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 256), nn.ReLU(inplace=True),
                                    nn.Dropout(dropout))
        self.se_head = nn.Linear(256, 1)
        self.sph_cyl_head = nn.Linear(256, 2)    # sphere, cylinder  (only used if multihead)
        self.axis_head = nn.Linear(256, 2)       # cos(2·axis), sin(2·axis) - axis is 180°-periodic

    def forward_head(self, feats):
        h = self.shared(feats)
        return torch.cat([self.se_head(h), self.sph_cyl_head(h), torch.tanh(self.axis_head(h))], 1)

    def forward(self, x):
        return self.forward_head(self.backbone(x))


def build_model(task: str, arch: str, pretrained=True, dropout=0.3, multihead=False):
    try:
        if task == "disease":
            return DiseaseNet(arch, pretrained=pretrained, dropout=dropout)
        return RefractiveNet(arch, pretrained=pretrained, dropout=dropout, multihead=multihead)
    except Exception as e:
        if pretrained and any(k in str(e).lower() for k in ("urlopen", "connection", "download", "resolve",
                                                            "huggingface", "offline", "hub")):
            raise SystemExit(f"Could not download ImageNet weights ({e}).\n"
                             "Check your internet connection, or run with --no-pretrained "
                             "(much worse results).") from e
        raise


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def model_filename(task: str, arch: str, tag: str = "") -> str:
    base = "disease" if task == "disease" else "refractive"
    return f"{base}_{arch}{('_' + tag) if tag else ''}"


def save_checkpoint(path: Path, model: nn.Module, meta: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    meta = ck["meta"]
    model = build_model(meta["task"], meta["arch"], pretrained=False,
                        dropout=meta.get("dropout", 0.3), multihead=meta.get("multihead", False))
    model.load_state_dict(ck["state_dict"])
    # sidecar JSON may hold newer metadata (e.g. test metrics written by evaluate.py)
    side = Path(path).with_suffix(".json")
    if side.exists():
        try:
            meta = {**meta, **json.loads(side.read_text())}
        except Exception:
            pass
    return model.to(device).eval(), meta


def new_model_version(task: str, arch: str, dataset: str) -> str:
    return f"{task[:3]}-{arch}-{dataset}-{dt.datetime.now().strftime('%Y%m%d-%H%M')}"


def find_checkpoints(task: str, include_demo=False):
    prefix = "disease_" if task == "disease" else "refractive_"
    found = sorted(config.MODELS_DIR.glob(f"{prefix}*.pth"))
    if include_demo:
        found += sorted(config.DEMO_MODELS_DIR.glob(f"{prefix}*.pth"))
    return found


def pick_best_checkpoint(task: str, allow_demo=True):
    """Best real model by validation metric; falls back to the synthetic demo model."""
    def score(p):
        try:
            m = json.loads(p.with_suffix(".json").read_text())
            v = m.get("val_metrics", {})
            return v.get("macro_roc_auc", 0) if task == "disease" else -v.get("mae", 1e9)
        except Exception:
            return -1e9
    real = find_checkpoints(task)
    if real:
        return max(real, key=score), False
    if allow_demo:
        demo = sorted(config.DEMO_MODELS_DIR.glob(f"{'disease' if task == 'disease' else 'refractive'}_*.pth"))
        if demo:
            return demo[0], True
    return None, False
