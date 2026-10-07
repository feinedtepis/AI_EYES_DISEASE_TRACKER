"""
Generate a SYNTHETIC fundus-like dataset for pipeline testing and UI demos.

    python scripts/make_synthetic_data.py --patients 400

These are cartoon images drawn with OpenCV (disc, cup, vessels, lesions,
tessellation, haze). They are NOT real retinas; models trained on them have
no medical meaning. They exist so that every script (train / evaluate /
baselines / predict / web app) can be run end-to-end before the real
datasets are downloaded, and so the GitHub repo does not redistribute
restricted medical images.

Writes:
  data/synthetic/images/*.png
  data/synthetic/disease_labels.csv      (per-eye multi-label disease labels)
  data/synthetic/refractive_labels.csv   (image_path,patient_id,age,sex,sphere,cylinder,axis,spherical_equivalent)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DISEASE_CODES  # noqa: E402

W, H, R = 460, 384, 178


def _smooth_noise(rng, shape, scale):
    small = rng.random((max(2, shape[0] // scale), max(2, shape[1] // scale))).astype(np.float32)
    return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)


def _vessel(rng, img, start, angle, length, width, color, tortuosity=0.08, steps=40):
    pts = [np.array(start, float)]
    a = angle
    for _ in range(steps):
        a += rng.normal(0, tortuosity)
        pts.append(pts[-1] + length / steps * np.array([np.cos(a), np.sin(a)]))
    for i in range(len(pts) - 1):
        w = max(1, int(round(width * (1 - 0.6 * i / steps))))
        cv2.line(img, tuple(map(int, pts[i])), tuple(map(int, pts[i + 1])), color, w, cv2.LINE_AA)
    return pts


def render_fundus(rng, eye="R", labels=None, se=-1.0, age=50, pigment=None):
    labels = labels or {}
    cx, cy = W // 2 + rng.integers(-6, 7), H // 2 + rng.integers(-5, 6)
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / R
    pig = pigment if pigment is not None else rng.uniform(0.75, 1.15)
    myopic = np.clip((-se - 1.0) / 7.0, 0, 1)            # drives tessellation / disc changes
    base = np.array([205, 92, 42], np.float32) * pig
    base[1] += 22 * myopic                                  # myopic fundi look paler / more orange
    img = np.ones((H, W, 3), np.float32) * base
    img *= (1.05 - 0.45 * d ** 2)[..., None]                # vignetting
    img *= (0.92 + 0.16 * _smooth_noise(rng, (H, W), 40))[..., None]

    # choroidal tessellation (stronger with more myopia / pathological myopia)
    tess = np.clip(myopic + (0.45 if labels.get("M") else 0) + rng.normal(0, 0.08), 0, 1)
    if tess > 0.05:
        n = _smooth_noise(rng, (H, W), 9)
        stripes = (np.sin(n * 28) > 0.55).astype(np.float32)
        stripes = cv2.GaussianBlur(stripes, (0, 0), 1.6)
        img *= (1 - 0.32 * tess * stripes)[..., None]

    disc_side = 1 if eye == "R" else -1
    dx, dy = int(cx + disc_side * 0.40 * R), int(cy - 0.04 * R)
    disc_r = int(0.12 * R * (1 + 0.15 * myopic))
    macula = (int(cx - disc_side * 0.08 * R), int(cy + 0.02 * R))
    # macula + fovea
    mac = np.exp(-((xx - macula[0]) ** 2 + (yy - macula[1]) ** 2) / (2 * (0.13 * R) ** 2))
    img *= (1 - 0.35 * mac)[..., None]
    cv2.circle(img, macula, 2, (230, 180, 120), -1, cv2.LINE_AA)

    # peripapillary atrophy crescent (myopia)
    if labels.get("M") or myopic > 0.6:
        cv2.ellipse(img, (dx - disc_side * 4, dy), (int(disc_r * 1.45), int(disc_r * 1.2)), 0, 0, 360,
                    (235, 200, 160), -1, cv2.LINE_AA)
    # optic disc and cup
    cv2.ellipse(img, (dx, dy), (disc_r, int(disc_r * (1.08 + 0.15 * myopic))), 0, 0, 360, (238, 196, 132), -1, cv2.LINE_AA)
    cdr = rng.uniform(0.62, 0.85) if labels.get("G") else rng.uniform(0.2, 0.42)
    cv2.ellipse(img, (dx, dy), (int(disc_r * cdr), int(disc_r * cdr * 1.05)), 0, 0, 360, (252, 236, 200), -1, cv2.LINE_AA)

    # vessels (arteries narrower & straighter in hypertension)
    art_w = 2.2 if labels.get("H") else 3.4
    vcol, acol = (110, 18, 18), (165, 38, 28)
    temporal = np.pi if disc_side > 0 else 0.0          # direction from the disc towards the macula
    nasal = 0.0 if disc_side > 0 else np.pi
    branches = [(temporal + s * off, 1.15) for s in (-1, 1) for off in (0.7, 1.25)] + \
               [(nasal + s * 0.7, 0.45) for s in (-1, 1)]
    for a0, length in branches:
        # temporal arcades curve back around the macula
        curl = 0.0
        _vessel(rng, img, (dx, dy), a0 + rng.normal(0, 0.08) + curl, rng.uniform(0.85, 1.1) * length * R, 4.4, vcol)
        _vessel(rng, img, (dx, dy), a0 + rng.normal(0, 0.12), rng.uniform(0.8, 1.05) * length * R, art_w, acol,
                tortuosity=0.04 if labels.get("H") else 0.08)

    def rand_point(center=None, spread=0.55):
        while True:
            c = center or (cx, cy)
            p = (int(c[0] + rng.normal(0, spread * R)), int(c[1] + rng.normal(0, spread * R)))
            if (p[0] - cx) ** 2 + (p[1] - cy) ** 2 < (0.9 * R) ** 2:
                return p

    if labels.get("D"):     # haemorrhages / microaneurysms + hard exudates
        for _ in range(rng.integers(12, 45)):
            cv2.circle(img, rand_point(), int(rng.integers(1, 5)), (120, 10, 10), -1, cv2.LINE_AA)
        for _ in range(rng.integers(6, 30)):
            cv2.circle(img, rand_point(macula, 0.3), int(rng.integers(1, 4)), (245, 222, 120), -1, cv2.LINE_AA)
    if labels.get("A"):     # drusen clustered at the macula + pigment clumps
        for _ in range(rng.integers(15, 45)):
            cv2.circle(img, rand_point(macula, 0.13), int(rng.integers(1, 4)), (238, 210, 130), -1, cv2.LINE_AA)
        for _ in range(rng.integers(2, 7)):
            cv2.circle(img, rand_point(macula, 0.1), int(rng.integers(2, 5)), (60, 25, 15), -1, cv2.LINE_AA)
    if labels.get("H"):     # cotton-wool spots + flame haemorrhages
        for _ in range(rng.integers(2, 6)):
            p = rand_point((dx, dy), 0.35)
            cv2.ellipse(img, p, (int(rng.integers(5, 10)), int(rng.integers(3, 7))), float(rng.uniform(0, 180)), 0, 360,
                        (235, 225, 215), -1, cv2.LINE_AA)
        for _ in range(rng.integers(2, 6)):
            p = rand_point((dx, dy), 0.4)
            cv2.ellipse(img, p, (int(rng.integers(6, 12)), 2), float(rng.uniform(0, 180)), 0, 360, (130, 15, 12), -1)
    if labels.get("O"):     # e.g. laser scars or a pale patch
        if rng.random() < 0.5:
            c = rand_point(None, 0.4)
            for i in range(-3, 4):
                for j in range(-3, 4):
                    cv2.circle(img, (c[0] + 9 * i, c[1] + 9 * j), 2, (70, 40, 25), -1, cv2.LINE_AA)
        else:
            cv2.ellipse(img, rand_point(None, 0.45), (int(rng.integers(14, 30)), int(rng.integers(10, 20))),
                        float(rng.uniform(0, 180)), 0, 360, (225, 185, 150), -1, cv2.LINE_AA)

    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    if labels.get("C"):     # cataract: blur + haze + lower contrast
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(2.5, 5.0))
        haze = rng.uniform(0.25, 0.45)
        img = img * (1 - haze) + np.array([205, 185, 160], np.float32) * haze * 0.8
    img += rng.normal(0, 3.0, img.shape).astype(np.float32)
    circle = (d <= 1.0).astype(np.float32)
    circle = cv2.GaussianBlur(circle, (0, 0), 1.2)
    img = np.clip(img * circle[..., None], 0, 255).astype(np.uint8)
    cv2.putText(img, "SYNTHETIC", (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (70, 70, 70), 1, cv2.LINE_AA)
    return img


def save(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    rows_d, rows_r = [], []
    for pid in range(a.patients):
        age = int(rng.integers(20, 86))
        sex = "M" if rng.random() < 0.5 else "F"
        prev = {"D": 0.18, "G": 0.10, "C": 0.05 + 0.25 * (age > 60), "A": 0.02 + 0.18 * (age > 60),
                "H": 0.10, "M": 0.12, "O": 0.10}
        pat = {c: rng.random() < p for c, p in prev.items()}
        se_base = rng.normal(-8.0, 2.5) if pat["M"] else float(np.clip(rng.normal(-1.0, 1.8), -6.0, 4.0))
        pig = rng.uniform(0.75, 1.15)
        for eye in ("L", "R"):
            lab = {c: int(v and (rng.random() < 0.8)) for c, v in pat.items()}
            if pat["M"]:
                lab["M"] = 1
            lab["N"] = int(not any(lab.values()))
            se = se_base + rng.normal(0, 0.4)
            cyl = -round(abs(rng.normal(0, 0.6)) * 4) / 4
            sph = round((se - cyl / 2) * 4) / 4
            se = sph + cyl / 2
            axis = int(rng.integers(0, 180))
            img = render_fundus(rng, eye, lab, se, age, pig)
            p = out / "images" / f"syn_{pid:04d}_{eye}.png"
            save(img, p)
            rows_d.append({"image_path": str(p), "patient_id": f"syn_{pid:04d}", "eye": eye, "age": age, "sex": sex,
                           "source": "synthetic", **{c: lab[c] for c in DISEASE_CODES}})
            rows_r.append({"image_path": str(p), "patient_id": f"syn_{pid:04d}", "eye": eye, "age": age, "sex": sex,
                           "sphere": sph, "cylinder": cyl, "axis": axis, "spherical_equivalent": se,
                           "source": "synthetic", **{c: lab[c] for c in DISEASE_CODES}})
        if (pid + 1) % 100 == 0:
            print(f"  {pid + 1}/{a.patients} patients")
    pd.DataFrame(rows_d).to_csv(out / "disease_labels.csv", index=False)
    pd.DataFrame(rows_r).to_csv(out / "refractive_labels.csv", index=False)
    print(f"Wrote {len(rows_d)} synthetic images to {out}")


if __name__ == "__main__":
    main()
