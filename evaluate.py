"""
Evaluation + report generation.

    python evaluate.py --task disease                      # best disease model, held-out TEST patients
    python evaluate.py --task disease --checkpoint models/disease_resnet50.pth
    python evaluate.py --task disease --external rfmid --data-root data/RFMiD   # generalisation check
    python evaluate.py --task refractive
    python evaluate.py --task disease --compare            # table of all trained models + baselines

Outputs (reports/<model name>/): metrics.json, per_class.csv, subgroups.csv,
ROC / PR / confusion plots (or scatter / residual plots) and evaluation_report.pdf.
Thresholds come from the VALIDATION set; the test set is never used for tuning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

# fixed categorical order (validated colour-blind-safe reference palette); one hue per class, never cycled
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"axes.prop_cycle": cycler(color=PALETTE), "lines.linewidth": 1.8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa5ab",
                     "axes.labelcolor": "#2b3a42", "xtick.color": "#52606a", "ytick.color": "#52606a",
                     "axes.titleweight": "bold", "font.size": 9.5, "legend.frameon": False})
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

import config
from config import DISEASE_CODES, SHORT_NAMES, TrainConfig
from dataset import FundusDataset, get_split, load_index
from metrics import age_group, multilabel_metrics, regression_metrics
from models import load_model, pick_best_checkpoint, find_checkpoints
import report_pdf as R


def run_model(model, df, task, meta, device, batch_size=32, num_workers=2):
    ds = FundusDataset(df, task, int(meta["img_size"]), False, True,
                       meta.get("target_mean", 0.0), meta.get("target_std", 1.0))
    dl = DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers)
    outs = []
    model.eval()
    with torch.no_grad():
        for x, _, _, _ in dl:
            outs.append(model(x.to(device)).float().cpu())
    out = torch.cat(outs).numpy()
    if task == "disease":
        return 1 / (1 + np.exp(-out))
    return out[:, 0] * meta.get("target_std", 1.0) + meta.get("target_mean", 0.0)


# ---------------------------------------------------------------------------
# Disease plots
# ---------------------------------------------------------------------------
def plot_roc_pr(y, p, m, out_dir, demo):
    fig_r, ax_r = plt.subplots(figsize=(7.4, 4.6))
    fig_p, ax_p = plt.subplots(figsize=(7.4, 4.6))
    for k, c in enumerate(DISEASE_CODES):
        keep = m[:, k] > 0
        yy, pp = y[keep, k], p[keep, k]
        if 0 < yy.sum() < len(yy):
            fpr, tpr, _ = roc_curve(yy, pp)
            prec, rec, _ = precision_recall_curve(yy, pp)
            from sklearn.metrics import auc, average_precision_score
            ax_r.plot(fpr, tpr, label=f"{SHORT_NAMES[c]} (AUC {auc(fpr, tpr):.2f}, n+={int(yy.sum())})")
            ax_p.plot(rec, prec, label=f"{SHORT_NAMES[c]} (AP {average_precision_score(yy, pp):.2f}, prev {yy.mean():.2f})")
    ax_r.plot([0, 1], [0, 1], "--", color="#999", lw=0.8)
    ax_r.set(xlabel="1 - specificity (FPR)", ylabel="Sensitivity (TPR)", title="ROC curves (test)")
    ax_p.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-Recall curves (test)")
    for ax, fig, name in ((ax_r, fig_r, "roc.png"), (ax_p, fig_p, "pr.png")):
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
        ax.grid(alpha=0.25)
        if demo:
            ax.text(0.5, 0.5, "SYNTHETIC DATA\nPIPELINE TEST ONLY", transform=ax.transAxes, ha="center",
                    va="center", fontsize=22, color="red", alpha=0.18, rotation=25)
        fig.tight_layout(); fig.savefig(out_dir / name, dpi=140); plt.close(fig)


def plot_confusions(per_class, out_dir, demo):
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for ax, c in zip(axes.ravel(), DISEASE_CODES):
        pc = per_class[c]
        mat = np.array([[pc["tn"], pc["fp"]], [pc["fn"], pc["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["pred -", "pred +"], fontsize=7)
        ax.set_yticks([0, 1], ["true -", "true +"], fontsize=7)
        ax.set_title(f"{SHORT_NAMES[c]} (thr {pc['threshold']:.2f})", fontsize=9)
    fig.suptitle("Per-class confusion matrices (multi-label, one-vs-rest)" +
                 ("  —  SYNTHETIC PIPELINE TEST" if demo else ""), fontsize=10)
    fig.tight_layout(); fig.savefig(out_dir / "confusion.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Refractive plots
# ---------------------------------------------------------------------------
def plot_regression(y, yhat, out_dir, demo):
    err = yhat - y
    lo, hi = min(y.min(), yhat.min()) - 0.5, max(y.max(), yhat.max()) + 0.5
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].scatter(y, yhat, s=10, alpha=0.6)
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8, label="perfect")
    ax[0].fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="#2a7", alpha=0.12, label="±1.00 D")
    ax[0].set(xlabel="Measured SE (D)", ylabel="Predicted SE (D)", title="Predicted vs measured", xlim=(lo, hi), ylim=(lo, hi))
    ax[0].legend(fontsize=8)
    ax[1].scatter(y, err, s=10, alpha=0.6, color="#c55")
    ax[1].axhline(0, color="k", lw=0.8)
    for v in (-1, 1):
        ax[1].axhline(v, color="#2a7", lw=0.8, ls="--")
    ax[1].axhline(err.mean(), color="#555", lw=0.8, ls=":", label=f"mean bias {err.mean():+.2f} D")
    ax[1].set(xlabel="Measured SE (D)", ylabel="Prediction - measured (D)", title="Residuals")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.25)
        if demo:
            a.text(0.5, 0.5, "SYNTHETIC DATA\nPIPELINE TEST ONLY", transform=a.transAxes, ha="center",
                   va="center", fontsize=18, color="red", alpha=0.18, rotation=25)
    fig.tight_layout(); fig.savefig(out_dir / "regression.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Subgroups
# ---------------------------------------------------------------------------
def subgroup_table(df, task, preds, meta):
    df = df.copy().reset_index(drop=True)
    df["age_group"] = df["age"].map(age_group)
    groups = [("age_group", "Age group"), ("sex", "Sex"), ("eye", "Eye"), ("source", "Source dataset")]
    rows = []
    if task == "refractive":
        df["se_band"] = pd.cut(df["spherical_equivalent"], [-99, -6, -3, -0.5, 0.5, 99],
                               labels=["high myopia (<= -6)", "moderate (-6..-3)", "low myopia (-3..-0.5)",
                                       "emmetropia (±0.5)", "hyperopia (> +0.5)"])
        groups.append(("se_band", "Refraction band"))
        for c in DISEASE_CODES:                       # disease category, if the CSV has those columns
            if c in df.columns and df[c].notna().any():
                df[f"has_{c}"] = df[c].map({1: f"{SHORT_NAMES[c]}+", 0: f"{SHORT_NAMES[c]}-"})
                groups.append((f"has_{c}", f"Disease: {SHORT_NAMES[c]}"))
    else:
        th = meta.get("thresholds", [0.5] * len(DISEASE_CODES))
    for col, label in groups:
        if col not in df.columns:
            continue
        for val, g in df.groupby(col, observed=True):
            idx = g.index.to_numpy()
            if task == "disease":
                y = g[DISEASE_CODES].to_numpy(float)
                m = (~np.isnan(y)).astype(float)
                res = multilabel_metrics(np.nan_to_num(y), preds[idx], m, th, DISEASE_CODES, ci=False)["summary"]
                rows.append({"variable": label, "group": str(val), "n_images": len(g),
                             "n_patients": g.patient_id.nunique(), "macro_roc_auc": res["macro_roc_auc"],
                             "macro_f1": res["macro_f1"], "macro_sensitivity": res["macro_recall"],
                             "macro_specificity": res["macro_specificity"],
                             "small_sample": len(g) < 50})
            else:
                res = regression_metrics(g["spherical_equivalent"].to_numpy(), preds[idx], ci=False)
                rows.append({"variable": label, "group": str(val), "n_images": len(g),
                             "n_patients": g.patient_id.nunique(), "mae": res["mae"], "rmse": res["rmse"],
                             "bias": res["bias_mean_error"], "within_0.50D": res["within_0.50D"],
                             "within_1.00D": res["within_1.00D"], "small_sample": len(g) < 30})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def write_pdf(out_dir, task, meta, res, sub, baselines, external, demo, split_label):
    story = [R.Paragraph(f"EyeVision AI — {'Disease classifier' if task == 'disease' else 'Refractive-error regressor'} "
                         f"evaluation", R.H1),
             R.Paragraph(f"Model <b>{meta.get('model_version')}</b> ({meta.get('arch')}), trained "
                         f"{meta.get('train_date')} on <b>{meta.get('dataset')}</b>. Evaluated on: {split_label}.", R.BODY)]
    if demo:
        story.append(R.Paragraph("<b>SYNTHETIC DATA — PIPELINE TEST ONLY.</b> This model was trained and tested on "
                                 "artificially generated images. The numbers below verify that the code runs; they "
                                 "say nothing about real-world or clinical performance.", R.WARN))
    story.append(R.Paragraph(config.DISCLAIMER, R.SMALL))
    if task == "disease":
        s, pc = res["summary"], res["per_class"]
        story += [R.Paragraph("Summary (macro = mean over classes; micro = pooled over all labels)", R.H2),
                  R.table([["Metric", "Macro", "Micro"],
                           ["ROC-AUC", R.fmt(s["macro_roc_auc"]), R.fmt(s["micro_roc_auc"])],
                           ["PR-AUC", R.fmt(s["macro_pr_auc"]), R.fmt(s["micro_pr_auc"])],
                           ["F1", R.fmt(s["macro_f1"]), R.fmt(s["micro_f1"])],
                           ["Precision", R.fmt(s["macro_precision"]), R.fmt(s["micro_precision"])],
                           ["Sensitivity (recall)", R.fmt(s["macro_recall"]), R.fmt(s["micro_recall"])],
                           ["Specificity", R.fmt(s["macro_specificity"]), R.fmt(s["micro_specificity"])]],
                          [170, 100, 100]),
                  R.Paragraph("Per-class metrics (threshold tuned on validation set)", R.H2)]
        rows = [["Class", "n", "n+", "Thr", "Prec", "Sens", "Spec", "F1", "ROC-AUC (95% CI)", "PR-AUC"]]
        for c in DISEASE_CODES:
            v = pc[c]
            rows.append([SHORT_NAMES[c] + (" *" if v["small_sample"] else ""), v["n"], v["n_pos"], R.fmt(v["threshold"], nd=2),
                         R.fmt(v["precision"]), R.fmt(v["recall_sensitivity"]), R.fmt(v["specificity"]),
                         R.fmt(v["f1"]), f"{R.fmt(v['roc_auc'])} {R.fmt(v['roc_auc_ci95'], nd=2)}", R.fmt(v["pr_auc"])])
        story += [R.table(rows), R.Paragraph("* fewer than 30 positive images — metric is unstable, interpret with caution. "
                                             "PR-AUC should be compared with the class prevalence (a random model's PR-AUC "
                                             "equals the prevalence).", R.SMALL),
                  R.image(out_dir / "roc.png", 12.5), R.image(out_dir / "pr.png", 12.5),
                  R.image(out_dir / "confusion.png", 17)]
    else:
        s = res
        story += [R.Paragraph("Spherical-equivalent regression", R.H2),
                  R.table([["n", "MAE (D)", "MAE 95% CI", "RMSE (D)", "MSE", "R²", "Bias (D)", "±0.50 D", "±1.00 D", "±2.00 D"],
                           [s["n"], R.fmt(s["mae"], nd=2), R.fmt(s.get("mae_ci95"), nd=2), R.fmt(s["rmse"], nd=2),
                            R.fmt(s["mse"], nd=2), R.fmt(s["r2"]), R.fmt(s["bias_mean_error"], nd=2),
                            R.fmt(s["within_0.50D"], True), R.fmt(s["within_1.00D"], True), R.fmt(s["within_2.00D"], True)]]),
                  R.Spacer(1, 6), R.image(out_dir / "regression.png", 17),
                  R.Paragraph("An estimated SE is not an eyeglass prescription; subjective refraction by a "
                              "professional is required.", R.SMALL)]
    if baselines:
        story.append(R.Paragraph("Comparison with simple baselines (same test patients)", R.H2))
        if task == "disease":
            rows = [["Model", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1"]]
            rows.append([f"{meta.get('arch')} (this model)", R.fmt(res["summary"]["macro_roc_auc"]),
                         R.fmt(res["summary"]["macro_pr_auc"]), R.fmt(res["summary"]["macro_f1"])])
            for name, b in baselines.items():
                rows.append([name, R.fmt(b["summary"]["macro_roc_auc"]), R.fmt(b["summary"]["macro_pr_auc"]),
                             R.fmt(b["summary"]["macro_f1"])])
        else:
            rows = [["Model", "MAE (D)", "RMSE (D)", "R²", "±1.00 D"]]
            rows.append([f"{meta.get('arch')} (this model)", R.fmt(res["mae"], nd=2), R.fmt(res["rmse"], nd=2),
                         R.fmt(res["r2"]), R.fmt(res["within_1.00D"], True)])
            for name, b in baselines.items():
                rows.append([name, R.fmt(b["mae"], nd=2), R.fmt(b["rmse"], nd=2), R.fmt(b["r2"]),
                             R.fmt(b["within_1.00D"], True)])
        story.append(R.table(rows))
    if sub is not None and len(sub):
        story.append(R.Paragraph("Subgroup analysis (rows marked * are small; differences may be noise)", R.H2))
        cols = [c for c in sub.columns if c != "small_sample"]
        rows = [cols] + [[(str(r[c]) + (" *" if (c == "group" and r["small_sample"]) else "")) if not isinstance(r[c], float)
                          else R.fmt(r[c], c.startswith("within"), 3) for c in cols] for _, r in sub.iterrows()]
        story.append(R.table(rows, font=7))
    if external:
        story.append(R.Paragraph("External dataset (generalisation check)", R.H2))
        s = external["summary"]
        story.append(R.Paragraph(f"{external['name']}: {s['n_images']} images. Macro ROC-AUC "
                                 f"{R.fmt(s['macro_roc_auc'])}, macro PR-AUC {R.fmt(s['macro_pr_auc'])}, macro F1 "
                                 f"{R.fmt(s['macro_f1'])} over classes {', '.join(s['classes_in_macro'])}. Classes "
                                 f"without an equivalent label in this dataset are excluded (masked).", R.BODY))
    R.build_pdf(out_dir / "evaluation_report.pdf", story, "EyeVision AI evaluation")


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return _json_safe(o.item())
    return o


def compare(task):
    rows = []
    prefix = "disease_" if task == "disease" else "refractive_"
    for p in sorted(config.MODELS_DIR.glob(f"{prefix}*.json")):   # metadata files: works even if a .pth was removed
        m = json.loads(p.read_text())
        t = m.get("test_metrics") or {}
        rows.append({"model": p.stem, "arch": m.get("arch"), "loss": m.get("loss"), "img": m.get("img_size"),
                     **({k: t.get(k) for k in ("macro_roc_auc", "macro_pr_auc", "macro_f1", "micro_f1")}
                        if task == "disease" else {k: t.get(k) for k in ("mae", "rmse", "r2", "within_0.50D", "within_1.00D")})})
    bfile = config.REPORTS_DIR / f"baselines_{task}.json"
    if bfile.exists():
        for name, b in json.loads(bfile.read_text())["results"].items():
            s = b["summary"] if task == "disease" else b
            rows.append({"model": f"baseline: {name}", **({k: s.get(k) for k in ("macro_roc_auc", "macro_pr_auc", "macro_f1", "micro_f1")}
                                                           if task == "disease" else {k: s.get(k) for k in ("mae", "rmse", "r2", "within_0.50D", "within_1.00D")})})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No trained models found."); return
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    out = config.REPORTS_DIR / f"comparison_{task}.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  (test metrics come from each model's last evaluate.py run)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["disease", "refractive"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--external", default="", help="evaluate on a whole external dataset, e.g. rfmid")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--data-csv", default="")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.compare:
        return compare(a.task)

    device = config.resolve_device(a.device)
    ckpt = Path(a.checkpoint) if a.checkpoint else pick_best_checkpoint(a.task, allow_demo=False)[0]
    if ckpt is None:
        raise SystemExit("No trained model in models/. Train one first (python train.py --task ...).")
    model, meta = load_model(ckpt, device)
    demo = bool(meta.get("is_demo"))
    out_dir = config.REPORTS_DIR / ckpt.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {ckpt.name} ({meta.get('model_version')}) on {device}")

    cfg = TrainConfig(task=a.task, seed=meta.get("seed", 42))
    if a.external:
        df = load_index(a.task, a.external, a.data_root, a.data_csv)
        split_label = f"entire external dataset '{a.external}'"
    else:
        df = load_index(a.task, meta["dataset"], a.data_root, a.data_csv or meta.get("data_csv", ""))
        df = get_split(df, a.task, meta.get("split_name", meta["dataset"]), cfg)
        df = df[df.split == a.split]
        split_label = f"held-out {a.split.upper()} split ({df.patient_id.nunique()} patients, {len(df)} images; split by patient)"
    df = df.reset_index(drop=True)
    preds = run_model(model, df, a.task, meta, device, a.batch_size, a.num_workers)
    pd.DataFrame({"image_path": df.image_path, **({f"p_{c}": preds[:, k] for k, c in enumerate(DISEASE_CODES)}
                  if a.task == "disease" else {"se_true": df.spherical_equivalent, "se_pred": preds})}
                 ).to_csv(out_dir / f"predictions_{a.external or a.split}.csv", index=False)

    baselines = {}
    bfile = config.REPORTS_DIR / f"baselines_{a.task}.json"
    if bfile.exists() and not a.external:
        baselines = json.loads(bfile.read_text())["results"]

    if a.task == "disease":
        y = df[DISEASE_CODES].to_numpy(float)
        m = (~np.isnan(y)).astype(float)
        y = np.nan_to_num(y)
        res = multilabel_metrics(y, preds, m, meta["thresholds"], DISEASE_CODES)
        plot_roc_pr(y, preds, m, out_dir, demo)
        plot_confusions(res["per_class"], out_dir, demo)
        pd.DataFrame(res["per_class"]).T.to_csv(out_dir / "per_class.csv")
        s = res["summary"]
        print(f"\nMacro ROC-AUC {s['macro_roc_auc']:.3f} | macro PR-AUC {s['macro_pr_auc']:.3f} | "
              f"macro F1 {s['macro_f1']:.3f} | micro F1 {s['micro_f1']:.3f}")
        print(pd.DataFrame(res["per_class"]).T[["n_pos", "precision", "recall_sensitivity", "specificity",
                                                "f1", "roc_auc", "pr_auc"]].to_string(float_format=lambda v: f"{v:.3f}"))
        headline = s
    else:
        yt = df["spherical_equivalent"].to_numpy(float)
        res = regression_metrics(yt, preds)
        plot_regression(yt, preds, out_dir, demo)
        print(f"\nMAE {res['mae']:.2f} D | RMSE {res['rmse']:.2f} D | R² {res['r2']:.3f} | "
              f"±0.50 D {100 * res['within_0.50D']:.1f}% | ±1.00 D {100 * res['within_1.00D']:.1f}% | "
              f"±2.00 D {100 * res['within_2.00D']:.1f}%")
        headline = res

    sub = subgroup_table(df, a.task, preds, meta)
    sub.to_csv(out_dir / "subgroups.csv", index=False)
    external = None
    if a.external:
        external = {"name": a.external, **res} if a.task == "disease" else None
    write_pdf(out_dir, a.task, meta, res, sub, baselines, external, demo, split_label)
    (out_dir / "metrics.json").write_text(json.dumps(_json_safe({"model": ckpt.name, "split": split_label, **(
        res if a.task == "disease" else {"summary": res})}), indent=2))

    # store headline metrics next to the model (shown on the app's Model Information page)
    side = ckpt.with_suffix(".json")
    sm = json.loads(side.read_text())
    if a.external:
        sm.setdefault("external_metrics", {})[a.external] = _json_safe(headline)
    elif a.split == "test":
        sm["test_metrics"] = _json_safe(headline)
        if a.task == "disease":
            sm["test_per_class"] = _json_safe(res["per_class"])
    side.write_text(json.dumps(_json_safe(sm), indent=2, default=str))
    print(f"\nReport: {out_dir / 'evaluation_report.pdf'}")
    if demo:
        print("NOTE: synthetic demo model - these numbers only show that the pipeline runs.")


if __name__ == "__main__":
    main()
