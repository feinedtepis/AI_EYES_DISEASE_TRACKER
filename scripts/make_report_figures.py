"""
Figures for the project report (reports/figures/*.png):
  architecture.png, dataset.png, model_comparison.png, gradcam_examples.png, synthetic_examples.png

    python scripts/make_report_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from config import DISEASE_CODES, SHORT_NAMES  # noqa: E402

OUT = config.REPORTS_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)
INK, MUTED, LINE = "#15242e", "#5d6d77", "#c9d3d8"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9aa5ab", "xtick.color": "#52606a", "ytick.color": "#52606a",
                     "axes.labelcolor": "#2b3a42", "legend.frameon": False})


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#ffffff", ec="#0e6b70"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.3))
        ax.text(x + w / 2, y + h - 2.3, title, ha="center", va="top", fontsize=9.2, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.1, sub, ha="center", va="top", fontsize=7.4, color=MUTED, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color="#5d6d77", lw=1.1))

    box(1, 26, 15, 16, "Upload", "left / right fundus\nphoto + patient ID,\nage, sex, date", ec="#5d6d77")
    box(21, 26, 16, 16, "Preprocess", "retina detection,\ncrop black borders,\nresize, quality check")
    box(42, 33, 18, 11, "Disease model", "EfficientNet-B0 / ResNet-50\n8 sigmoid outputs")
    box(42, 18.5, 18, 11, "Refraction model", "ResNet-50 regression\nSE in diopters")
    box(65, 33, 15, 11, "Uncertainty", "MC dropout x10,\nval. thresholds")
    box(65, 18.5, 15, 11, "Grad-CAM", "attention map,\nresearch use only")
    box(84, 26, 15, 16, "SQLite", "patients +\nper-eye exams,\nmodel version", ec="#5d6d77")
    box(42, 1, 38, 12, "FastAPI + web UI", "dashboard · upload · results · history · trends · model info", fc="#eef6f6")
    box(1, 1, 36, 12, "Offline pipeline", "train.py · evaluate.py · baselines.py → models/*.pth + reports/", fc="#fbf6ee", ec="#c0541b")
    arrow(16.6, 34, 20.4, 34)
    arrow(37.6, 36, 41.4, 38); arrow(37.6, 32, 41.4, 24.5)
    arrow(60.6, 38.5, 64.4, 38.5); arrow(60.6, 24, 64.4, 24)
    arrow(60.6, 36, 64.4, 26)
    arrow(80.6, 38.5, 83.4, 36); arrow(80.6, 24, 83.4, 30)
    arrow(91.5, 25.4, 80.6, 9)
    arrow(37.6, 7, 41.4, 7)
    ax.text(19, 14.6, "trained checkpoints are loaded by the app", fontsize=7.2, color="#c0541b", ha="center")
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "architecture.png", dpi=170)
    plt.close(fig)


def dataset_fig():
    idx = config.DATA_DIR / "odir_index.csv"
    if not idx.exists():
        return
    df = pd.read_csv(idx)
    split = pd.read_csv(config.DATA_DIR / "splits" / "disease_odir_seed42.csv", dtype={"patient_id": str})
    df = df.merge(split, on="patient_id", how="left")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.5, 1]})
    counts = df[DISEASE_CODES].sum().astype(int)
    bars = ax[0].bar([SHORT_NAMES[c] if c != "H" else "Hyper-\ntension" for c in DISEASE_CODES], counts.values, color="#2a78d6", width=0.62)
    for b, v in zip(bars, counts.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 25, f"{v}", ha="center", fontsize=8, color=INK)
    ax[0].set_title("Images per label (per eye, ODIR-5K)", fontsize=10, loc="left", color=INK)
    ax[0].set_ylabel("eye images"); ax[0].tick_params(axis="x", labelsize=8.3)
    ax[0].grid(axis="y", color="#e6ecef"); ax[0].set_axisbelow(True)
    ages = df.drop_duplicates("patient_id")["age"].dropna()
    ax[1].hist(ages, bins=range(0, 100, 5), color="#1baf7a", edgecolor="white", linewidth=1.5)
    ax[1].set_title("Patient age distribution", fontsize=10, loc="left", color=INK)
    ax[1].set_xlabel("age (years)"); ax[1].set_ylabel("patients")
    ax[1].grid(axis="y", color="#e6ecef"); ax[1].set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "dataset.png", dpi=170); plt.close(fig)


def comparison_fig():
    rows = []
    for p in sorted(config.MODELS_DIR.glob("disease_*.json")):
        m = json.loads(p.read_text())
        t = m.get("test_metrics")
        if t:
            rows.append((m["arch"].replace("efficientnet_b0", "EfficientNet-B0").replace("resnet50", "ResNet-50")
                         + (" (partial FT)" if m.get("freeze_early") else ""), t))
    b = config.REPORTS_DIR / "baselines_disease.json"
    if b.exists():
        for name, r in json.loads(b.read_text())["results"].items():
            rows.append((name.replace(" (features)", "\n(hand-crafted)"), r["summary"]))
    if not rows:
        return
    metrics = [("macro_roc_auc", "Macro ROC-AUC"), ("macro_pr_auc", "Macro PR-AUC"), ("macro_f1", "Macro F1")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(rows)
    x = np.arange(len(metrics))
    for i, (name, t) in enumerate(rows):
        vals = [t.get(k) or 0 for k, _ in metrics]
        bars = ax.bar(x + (i - (len(rows) - 1) / 2) * w, vals, w * 0.92, color=PALETTE[i], label=name.replace("\n", " "))
        for bb, v in zip(bars, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.4, color=INK)
    ax.set_xticks(x, [m[1] for m in metrics]); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#e6ecef"); ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=len(rows), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Held-out test patients: deep models vs. baselines", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=170); plt.close(fig)


def gradcam_examples(n_per=1):
    """Grad-CAM of the best real model on real TEST images whose label is a disease."""
    from inference import EyeVisionEngine
    from gradcam import GradCAM, overlay_heatmap
    from dataset import build_transforms
    from preprocess import preprocess_fundus
    from PIL import Image
    import torch
    eng = EyeVisionEngine(allow_demo=False)
    if eng.disease is None:
        return
    idx = pd.read_csv(config.DATA_DIR / "odir_index.csv")
    split = pd.read_csv(config.DATA_DIR / "splits" / "disease_odir_seed42.csv", dtype={"patient_id": str})
    idx = idx.merge(split, on="patient_id")
    test = idx[idx.split == "test"]
    size = int(eng.disease_meta["img_size"])
    picks = []
    for c in ["D", "C", "G", "M", "A", "H"]:
        cand = test[(test[c] == 1) & (test[DISEASE_CODES].sum(1) == 1)]
        if len(cand):
            picks.append((c, cand.sample(1, random_state=3).iloc[0]))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.05 * len(picks), 4.8))
    cam_engine = GradCAM(eng.disease)
    for j, (c, r) in enumerate(picks):
        sq, mask, _ = preprocess_fundus(r["image_path"], size)
        x = build_transforms(size, False)(Image.fromarray(sq)).unsqueeze(0)
        with torch.no_grad():
            p = torch.sigmoid(eng.disease(x))[0].numpy()
        k = DISEASE_CODES.index(c)
        cam = cam_engine(x, k)
        axes[0, j].imshow(sq); axes[1, j].imshow(overlay_heatmap(sq, cam, mask))
        axes[0, j].set_title(f"label: {SHORT_NAMES[c]}", fontsize=8.5, color=INK)
        axes[1, j].text(0.5, -0.04, f"model p({SHORT_NAMES[c]}) = {p[k]:.2f}", fontsize=8, color=MUTED,
                        ha="center", va="top", transform=axes[1, j].transAxes)
        for a in axes[:, j]:
            a.axis("off")
    cam_engine.remove()
    fig.suptitle("Grad-CAM model attention on held-out ODIR-5K test images — research use only", fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(OUT / "gradcam_examples.png", dpi=170); plt.close(fig)


def synthetic_examples():
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic_data import render_fundus
    rng = np.random.default_rng(4)
    items = [("Normal", {}, -0.5), ("DR", {"D": 1}, -1), ("Glaucoma", {"G": 1}, -1), ("Cataract", {"C": 1}, 0),
             ("AMD", {"A": 1}, -1), ("Hypertension", {"H": 1}, -1), ("Myopia (SE −9 D)", {"M": 1}, -9), ("Other", {"O": 1}, -1)]
    fig, axes = plt.subplots(1, len(items), figsize=(12, 1.9))
    for a, (name, lab, se) in zip(axes, items):
        a.imshow(render_fundus(rng, "R", lab, se)[:, 38:-38]); a.axis("off"); a.set_title(name, fontsize=8, color=INK)
    fig.suptitle("Synthetic cartoon images used only for pipeline testing and the refractive demo model", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "synthetic_examples.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    architecture(); print("architecture")
    dataset_fig(); print("dataset")
    comparison_fig(); print("comparison")
    synthetic_examples(); print("synthetic")
    gradcam_examples(); print("gradcam")
