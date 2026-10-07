"""
Central configuration for EyeVision AI (research / educational project).

Everything that other modules need to agree on lives here: label names,
paths, default hyper-parameters, safety wording.  Command-line flags in
train.py / evaluate.py override the defaults below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # pre-cropped / resized images
MODELS_DIR = ROOT / "models"
DEMO_MODELS_DIR = MODELS_DIR / "demo"   # synthetic-data demo models (NOT medical)
REPORTS_DIR = ROOT / "reports"
DB_PATH = Path(os.environ.get("EYEVISION_DB", ROOT / "eyevision.db"))
UPLOAD_DIR = ROOT / "app" / "uploads"
HEATMAP_DIR = ROOT / "app" / "heatmaps"

for _d in (DATA_DIR, CACHE_DIR, MODELS_DIR, REPORTS_DIR, UPLOAD_DIR, HEATMAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Label space (shared by ODIR-5K, RFMiD mapping and synthetic data)
# ---------------------------------------------------------------------------
# Order matters: model output index i == DISEASE_CODES[i]
DISEASE_CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
DISEASE_NAMES = {
    "N": "Normal / healthy",
    "D": "Diabetic retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertensive retinopathy",
    "M": "Myopia (pathological)",
    "O": "Other abnormality",
}
SHORT_NAMES = {
    "N": "Normal", "D": "DR", "G": "Glaucoma", "C": "Cataract",
    "A": "AMD", "H": "Hypertension", "M": "Myopia", "O": "Other",
}
NUM_DISEASES = len(DISEASE_CODES)

# ---------------------------------------------------------------------------
# Image normalisation (ImageNet statistics - we use ImageNet-pretrained CNNs)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Supported architectures
# ---------------------------------------------------------------------------
DISEASE_ARCHS = ["efficientnet_b0", "efficientnet_b2", "resnet50", "convnext_tiny"]
REFRACTIVE_ARCHS = ["resnet50", "efficientnet_b0", "efficientnet_b2", "convnext_tiny"]


@dataclass
class TrainConfig:
    task: str = "disease"                 # disease | refractive
    dataset: str = "odir"                 # odir | rfmid | odir+rfmid | csv | synthetic
    arch: str = "efficientnet_b0"
    img_size: int = 384                   # 224 for quick CPU runs, 384-512 on GPU
    batch_size: int = 16
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    loss: str = "wbce"                    # bce | wbce | focal  (disease)  /  smoothl1 | mse (refractive)
    focal_gamma: float = 2.0
    pretrained: bool = True
    freeze_epochs: int = 1                # epochs with frozen backbone (head warm-up)
    dropout: float = 0.3
    patience: int = 6                     # early stopping
    num_workers: int = 2
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    hflip: bool = True                    # flip L<->R anatomy; see report for rationale
    multihead: bool = False               # refractive: also predict sphere/cyl/axis
    max_samples: int = 0                  # 0 = all (debug: limit dataset size)
    device: str = "auto"
    data_csv: str = ""                    # refractive CSV / custom disease CSV
    data_root: str = ""                   # folder containing the dataset
    tag: str = ""                         # optional suffix for the saved model name

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Inference / uncertainty
# ---------------------------------------------------------------------------
MC_DROPOUT_PASSES = 10        # Monte-Carlo dropout forward passes at inference
SE_CONFIDENCE_TOLERANCE_D = 1.0   # "confidence" = P(|error| <= 1.0 D) under the
                                  # model's predictive distribution (see report)

# ---------------------------------------------------------------------------
# Safety wording - used by the API, the CLI and the web UI
# ---------------------------------------------------------------------------
DISCLAIMER = (
    "Research/educational AI estimate only. "
    "This system is not a medical diagnostic tool and does not provide a clinical "
    "diagnosis or eyeglass prescription. Please consult a qualified "
    "ophthalmologist/optometrist for diagnosis and refraction."
)
SE_NOTE = "Experimental AI estimate. Not a prescription. Professional refraction is required for an actual prescription."
GRADCAM_NOTE = "Model attention visualization — research use only. It does not prove the model is looking at the correct pathology."
TREND_NOTE = ("Changes in model predictions do not necessarily mean the patient's actual "
              "eyesight or eye health changed. Image quality, camera, lighting and model "
              "error can all change the output.")


def resolve_device(name: str = "auto"):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
