"""
Non-deep-learning baselines, evaluated on the SAME patient-level test split.

Disease   : hand-crafted colour/texture features -> Logistic Regression and Random Forest
            (one-vs-rest, class-balanced), thresholds tuned on validation.
Refractive: (1) predict the training-set mean SE; (2) linear regression on age +
            image features (if age is available).

    python baselines.py --task disease  --dataset odir
    python baselines.py --task refractive --data-csv data/refraction.csv
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from config import DISEASE_CODES, TrainConfig
from dataset import get_split, load_index, split_name_for
from metrics import multilabel_metrics, regression_metrics, tune_thresholds
from preprocess import preprocess_fundus


def image_features(path, size=256):
    """~70 simple features: RGB/HSV histograms + stats, sharpness, edge density, bright/dark spot ratios."""
    rgb, mask, _ = preprocess_fundus(path, size)
    inside = mask > 0
    if inside.sum() < 100:
        inside = np.ones_like(mask, bool)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    f = []
    for img in (rgb, hsv):
        for ch in range(3):
            v = img[..., ch][inside]
            f += list(np.histogram(v, bins=8, range=(0, 256))[0] / max(len(v), 1))
            f += [v.mean() / 255, v.std() / 255]
    g = rgb[..., 1]
    clahe = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    lap = cv2.Laplacian(g, cv2.CV_64F)
    edges = cv2.Canny(clahe, 40, 120)
    gi = g[inside].astype(float)
    f += [lap[inside].var() / 1000, edges[inside].mean() / 255,
          float((gi > np.percentile(gi, 99)).mean()), float((gi < np.percentile(gi, 1)).mean()),
          float(np.percentile(gi, 99) - np.percentile(gi, 50)) / 255]
    return np.array(f, np.float32)


def featurize(df, size):
    X = []
    for i, p in enumerate(df.image_path):
        X.append(image_features(p, size))
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{len(df)}")
    return np.stack(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["disease", "refractive"], required=True)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--data-csv", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    dataset = a.dataset or ("odir" if a.task == "disease" else "csv")
    cfg = TrainConfig(task=a.task, seed=a.seed)
    df = load_index(a.task, dataset, a.data_root, a.data_csv)
    split_name = split_name_for(a.task, dataset, a.data_csv)
    df = get_split(df, a.task, split_name, cfg)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"Extracting hand-crafted features: train {len(tr)}, val {len(va)}, test {len(te)} images ...")
    Xtr, Xva, Xte = featurize(tr, a.size), featurize(va, a.size), featurize(te, a.size)
    results = {}

    if a.task == "disease":
        def fit_eval(make, name):
            P_va, P_te = np.zeros((len(va), len(DISEASE_CODES))), np.zeros((len(te), len(DISEASE_CODES)))
            for k, c in enumerate(DISEASE_CODES):
                keep = tr[c].notna().to_numpy()
                y = tr.loc[keep, c].astype(int).to_numpy()
                if y.min() == y.max():
                    P_va[:, k] = P_te[:, k] = y[0]
                    continue
                clf = make().fit(Xtr[keep], y)
                P_va[:, k] = clf.predict_proba(Xva)[:, 1]
                P_te[:, k] = clf.predict_proba(Xte)[:, 1]
            yv, yt = va[DISEASE_CODES].to_numpy(float), te[DISEASE_CODES].to_numpy(float)
            mv, mt = (~np.isnan(yv)).astype(float), (~np.isnan(yt)).astype(float)
            th = tune_thresholds(np.nan_to_num(yv), P_va, mv)
            res = multilabel_metrics(np.nan_to_num(yt), P_te, mt, th, DISEASE_CODES)
            results[name] = res
            s = res["summary"]
            print(f"{name:<34} macro ROC-AUC {s['macro_roc_auc']:.3f} | macro PR-AUC {s['macro_pr_auc']:.3f} | macro F1 {s['macro_f1']:.3f}")

        fit_eval(lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
                 "Logistic regression (features)")
        fit_eval(lambda: RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=a.seed), "Random forest (features)")
    else:
        ytr, yte = tr.spherical_equivalent.to_numpy(float), te.spherical_equivalent.to_numpy(float)
        results["Mean-SE baseline"] = regression_metrics(yte, np.full_like(yte, ytr.mean()))
        age_ok = tr.age.notna().all() and te.age.notna().all()
        Ftr = np.c_[Xtr, tr.age.to_numpy(float)] if age_ok else Xtr
        Fte = np.c_[Xte, te.age.to_numpy(float)] if age_ok else Xte
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Ftr, ytr)
        results["Linear (ridge) on image features" + (" + age" if age_ok else "")] = regression_metrics(yte, reg.predict(Fte))
        for n, r in results.items():
            print(f"{n:<44} MAE {r['mae']:.2f} D | RMSE {r['rmse']:.2f} | R² {r['r2']:.3f} | ±1.00 D {100 * r['within_1.00D']:.1f}%")

    out = config.REPORTS_DIR / f"baselines_{a.task}.json"

    def safe(o):
        if isinstance(o, dict):
            return {k: safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [safe(v) for v in o]
        return None if isinstance(o, float) and np.isnan(o) else o
    out.write_text(json.dumps({"dataset": dataset, "split": split_name, "results": safe(results)}, indent=2))
    print(f"Saved {out}. evaluate.py will include these in its comparison table.")


if __name__ == "__main__":
    main()
