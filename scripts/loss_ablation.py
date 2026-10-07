"""
Loss-function ablation for the imbalanced multi-label problem:
plain BCE vs class-weighted BCE vs focal loss.

To keep it cheap enough for a CPU, every loss trains the SAME linear head on
the SAME frozen ImageNet EfficientNet-B0 features (a 'linear probe'), with the
same patient-level split, and thresholds tuned on validation. Differences are
therefore due to the loss only. (Full fine-tuning would change absolute numbers.)

    python scripts/loss_ablation.py --dataset odir --img-size 224
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from config import DISEASE_CODES, TrainConfig  # noqa: E402
from dataset import FundusDataset, get_split, load_index  # noqa: E402
from losses import MaskedBCE, MaskedFocal, compute_pos_weight  # noqa: E402
from metrics import multilabel_metrics, tune_thresholds  # noqa: E402
from models import build_backbone  # noqa: E402


@torch.no_grad()
def features(backbone, df, size):
    ds = FundusDataset(df, "disease", size, False)
    dl = torch.utils.data.DataLoader(ds, 64, shuffle=False, num_workers=1)
    out, ys, ms = [], [], []
    for x, y, m, _ in dl:
        out.append(backbone(x)); ys.append(y); ms.append(m)
    return torch.cat(out), torch.cat(ys), torch.cat(ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="odir")
    ap.add_argument("--arch", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cfg = TrainConfig(task="disease", seed=a.seed)
    df = get_split(load_index("disease", a.dataset), "disease", a.dataset, cfg)
    parts = {s: df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test")}
    bb, dim = build_backbone(a.arch, pretrained=True)
    bb.eval()
    print("Extracting frozen features ...")
    F = {s: features(bb, d, a.img_size) for s, d in parts.items()}
    pw = compute_pos_weight(parts["train"], DISEASE_CODES)
    losses = {"BCE (unweighted)": MaskedBCE(), "Weighted BCE (pos_weight = neg/pos)": MaskedBCE(pw),
              "Focal loss (gamma=2)": MaskedFocal(2.0), "Focal loss (gamma=2) + pos_weight": MaskedFocal(2.0, pw)}
    rows, results = [], {}
    for name, crit in losses.items():
        torch.manual_seed(a.seed)
        head = torch.nn.Linear(dim, len(DISEASE_CODES))
        opt = torch.optim.AdamW(head.parameters(), 1e-3, weight_decay=1e-3)
        Xtr, ytr, mtr = F["train"]
        best, best_state = -1, None
        for ep in range(a.epochs):
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 128):
                idx = perm[i:i + 128]
                opt.zero_grad(); crit(head(Xtr[idx]), ytr[idx], mtr[idx]).backward(); opt.step()
            with torch.no_grad():
                pv = torch.sigmoid(head(F["val"][0])).numpy()
            auc = multilabel_metrics(F["val"][1].numpy(), pv, F["val"][2].numpy(), [0.5] * 8, DISEASE_CODES, ci=False)["summary"]["macro_roc_auc"]
            if auc > best:
                best, best_state = auc, {k: v.clone() for k, v in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad():
            pv = torch.sigmoid(head(F["val"][0])).numpy()
            pt = torch.sigmoid(head(F["test"][0])).numpy()
        th = tune_thresholds(F["val"][1].numpy(), pv, F["val"][2].numpy())
        res = multilabel_metrics(F["test"][1].numpy(), pt, F["test"][2].numpy(), th, DISEASE_CODES)
        s = res["summary"]
        # recall on the rare classes is where the loss choice matters most
        rare = ["G", "C", "A", "H", "M"]
        rare_rec = float(np.nanmean([res["per_class"][c]["recall_sensitivity"] for c in rare]))
        rows.append({"loss": name, "macro_roc_auc": s["macro_roc_auc"], "macro_pr_auc": s["macro_pr_auc"],
                     "macro_f1": s["macro_f1"], "macro_sensitivity": s["macro_recall"],
                     "rare_class_sensitivity": rare_rec, "macro_specificity": s["macro_specificity"]})
        results[name] = res
        print(f"{name:<40} AUC {s['macro_roc_auc']:.3f}  PR-AUC {s['macro_pr_auc']:.3f}  F1 {s['macro_f1']:.3f}  "
              f"rare-class sens {rare_rec:.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(config.REPORTS_DIR / "loss_ablation.csv", index=False)

    def safe(o):
        if isinstance(o, dict):
            return {k: safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [safe(v) for v in o]
        return None if isinstance(o, float) and np.isnan(o) else o
    (config.REPORTS_DIR / "loss_ablation.json").write_text(json.dumps(
        {"setup": f"linear probe on frozen ImageNet {a.arch} features, {a.img_size}px, {a.epochs} epochs, "
                  f"best epoch by val macro ROC-AUC, thresholds max-F1 on val", "results": safe(results)}, indent=2))
    print("Saved reports/loss_ablation.csv")


if __name__ == "__main__":
    main()
