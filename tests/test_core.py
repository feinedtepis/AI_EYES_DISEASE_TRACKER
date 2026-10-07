"""Unit tests for the parts that must never silently break.   Run:  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DISEASE_CODES  # noqa: E402
from dataset import load_refractive_csv, odir_keywords_to_labels, patient_split  # noqa: E402
from losses import MaskedBCE, MaskedFocal  # noqa: E402
from metrics import multilabel_metrics, regression_metrics  # noqa: E402


def test_odir_keywords():
    lab, excl, _ = odir_keywords_to_labels("moderate non proliferative retinopathy，laser spot")
    assert lab["D"] == 1 and lab["O"] == 1 and lab["N"] == 0 and not excl
    lab, excl, _ = odir_keywords_to_labels("normal fundus")
    assert lab["N"] == 1 and sum(lab.values()) == 1 and not excl
    lab, excl, _ = odir_keywords_to_labels("lens dust，normal fundus")
    assert lab["N"] == 1 and not excl
    lab, excl, _ = odir_keywords_to_labels("pathological myopia")
    assert lab["M"] == 1
    lab, excl, _ = odir_keywords_to_labels("dry age-related macular degeneration")
    assert lab["A"] == 1
    lab, excl, _ = odir_keywords_to_labels("suspected glaucoma")
    assert lab["G"] == 1
    _, excl, _ = odir_keywords_to_labels("low image quality")
    assert excl
    _, excl, _ = odir_keywords_to_labels("lens dust")       # no diagnosis -> unknown -> excluded
    assert excl


def test_patient_split_has_no_leakage():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(300):
        lab = {c: int(rng.random() < 0.1) for c in DISEASE_CODES}
        for eye in "LR":
            rows.append({"patient_id": f"p{p}", "eye": eye, **lab})
    df = patient_split(pd.DataFrame(rows), "disease", 0.15, 0.15, 1)
    assert df.groupby("patient_id")["split"].nunique().max() == 1
    assert set(df.split) == {"train", "val", "test"}


def test_refractive_csv_computes_se(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    pd.DataFrame([{"image_path": "a.png", "patient_id": 1, "age": 20, "sex": "F", "sphere": -2.0, "cylinder": -1.0,
                   "axis": 90, "spherical_equivalent": None}]).to_csv(tmp_path / "r.csv", index=False)
    df = load_refractive_csv(str(tmp_path / "r.csv"))
    assert abs(df.spherical_equivalent.iloc[0] - (-2.5)) < 1e-9


def test_masked_losses_ignore_unknown_labels():
    logits = torch.tensor([[5.0, -5.0]])
    y = torch.tensor([[0.0, 1.0]])            # both "wrong" ...
    m = torch.tensor([[0.0, 0.0]])            # ... but masked
    assert MaskedBCE()(logits, y, m).item() == 0.0
    assert MaskedFocal()(logits, y, m).item() == 0.0


def test_metrics_perfect_and_regression():
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float)
    res = multilabel_metrics(y, y * 0.9 + 0.05, np.ones_like(y), [0.5, 0.5], ["a", "b"], ci=False)
    assert res["summary"]["macro_roc_auc"] == 1.0 and res["summary"]["macro_f1"] == 1.0
    r = regression_metrics([-2.0, -1.0, 0.0], [-2.25, -1.0, 0.75])
    assert abs(r["mae"] - (0.25 + 0 + 0.75) / 3) < 1e-9
    assert abs(r["within_0.50D"] - 2 / 3) < 1e-9
