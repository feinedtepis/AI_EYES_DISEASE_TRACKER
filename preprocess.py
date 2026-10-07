"""
Fundus image preprocessing + simple image-quality checks.

Steps (applied identically at training and inference time):
  1. Locate the circular retinal region (threshold + largest contour).
  2. Crop to it, removing black borders, and pad to a square.
  3. Resize to a fixed size.
  (4. Normalisation with ImageNet statistics happens in the torch transforms.)

CLI - cache a dataset index at a fixed size so CPU training is fast:
    python preprocess.py --index data/odir_index.csv --size 384
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_rgb(path_or_img) -> np.ndarray:
    """Return an RGB uint8 array from a path, PIL image or numpy array."""
    if isinstance(path_or_img, np.ndarray):
        img = path_or_img
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img[..., :3].astype(np.uint8)
    if isinstance(path_or_img, Image.Image):
        return np.array(path_or_img.convert("RGB"))
    data = np.fromfile(str(path_or_img), dtype=np.uint8)      # handles unicode paths on Windows
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path_or_img}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Retina detection / cropping
# ---------------------------------------------------------------------------
def retina_mask(rgb: np.ndarray, thresh: int | None = None) -> np.ndarray:
    """Binary mask of the (bright) retinal disc against the dark camera border."""
    # Red channel is the brightest channel in colour fundus photos; combine with gray.
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    chan = np.maximum(gray, rgb[..., 0])
    if thresh is None:
        # adaptive: 10 or 7% of a robust maximum, whichever is larger
        thresh = max(10, int(0.07 * np.percentile(chan, 99)))
    mask = (chan > thresh).astype(np.uint8)
    k = max(3, (min(rgb.shape[:2]) // 100) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    # keep the largest connected component (ignores text / watermarks in corners)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def crop_retina(rgb: np.ndarray, pad_frac: float = 0.02):
    """Crop the retinal region and pad to a square. Returns (square_rgb, square_mask, info)."""
    h, w = rgb.shape[:2]
    mask = retina_mask(rgb)
    area_frac = float(mask.mean())
    ys, xs = np.where(mask > 0)
    if area_frac < 0.05 or len(xs) == 0:
        # nothing that looks like a retina - fall back to a centre square crop
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        crop = rgb[y0:y0 + s, x0:x0 + s]
        return crop, np.ones(crop.shape[:2], np.uint8), {"retina_found": False, "area_frac": area_frac}
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = rgb[y0:y1, x0:x1]
    cmask = mask[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    side = int(max(ch, cw) * (1 + 2 * pad_frac))
    out = np.zeros((side, side, 3), np.uint8)
    omask = np.zeros((side, side), np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = crop
    omask[oy:oy + ch, ox:ox + cw] = cmask
    bbox_fill = float(cmask.mean())          # ~0.785 for a full circle, higher if top/bottom cut
    return out, omask, {"retina_found": True, "area_frac": area_frac, "bbox_fill": bbox_fill}


def preprocess_fundus(path_or_img, size: int = 384):
    """Full preprocessing: crop retina -> square -> resize. Returns (rgb uint8, mask uint8, info)."""
    rgb = load_rgb(path_or_img)
    sq, m, info = crop_retina(rgb)
    interp = cv2.INTER_AREA if sq.shape[0] > size else cv2.INTER_CUBIC
    sq = cv2.resize(sq, (size, size), interpolation=interp)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return sq, m, info


# ---------------------------------------------------------------------------
# Image quality (heuristic, NOT a validated gradability model)
# ---------------------------------------------------------------------------
def quality_check(path_or_img) -> dict:
    rgb = load_rgb(path_or_img)
    sq, m, info = preprocess_fundus(rgb, 512)
    inside = m > 0
    warnings = []
    if not info["retina_found"]:
        warnings.append("No circular retinal region detected - this may not be a fundus photograph.")
        inside = np.ones(m.shape, bool)
    px = sq[inside].astype(np.float32)
    mean_rgb = px.mean(0) if len(px) else np.zeros(3)
    brightness = float(px.mean()) if len(px) else 0.0
    green = sq[..., 1]
    lap = cv2.Laplacian(green, cv2.CV_64F)
    sharpness = float(lap[inside].var()) if inside.any() else 0.0
    contrast = float(green[inside].std()) if inside.any() else 0.0
    red_ratio = float(mean_rgb[0] / (mean_rgb[2] + 1.0))

    if info["retina_found"] and red_ratio < 1.15:
        warnings.append("Colour distribution is unusual for a colour fundus photograph.")
    if brightness < 30:
        warnings.append("Image is very dark (under-exposed).")
    if brightness > 200:
        warnings.append("Image is very bright (over-exposed).")
    if sharpness < 8:
        warnings.append("Image appears blurred. This can come from poor focus OR from media "
                        "opacity such as cataract - interpret with care.")
    if contrast < 8:
        warnings.append("Very low contrast inside the retinal region.")
    gradable = info["retina_found"] and brightness >= 30 and brightness <= 200 and contrast >= 8
    return {
        "gradable": bool(gradable),
        "warnings": warnings,
        "metrics": {
            "retina_found": bool(info["retina_found"]),
            "retina_area_fraction": round(float(info["area_frac"]), 3),
            "brightness": round(brightness, 1),
            "sharpness_laplacian_var": round(sharpness, 1),
            "contrast": round(contrast, 1),
            "red_blue_ratio": round(red_ratio, 2),
        },
        "note": "Heuristic quality check, not a validated image-gradability model.",
    }


# ---------------------------------------------------------------------------
# Dataset caching
# ---------------------------------------------------------------------------
def cache_path_for(src: str, size: int) -> Path:
    h = hashlib.md5(str(Path(src).resolve()).encode()).hexdigest()[:12]
    return config.CACHE_DIR / str(size) / f"{Path(src).stem}_{h}.png"


def cache_index(index_csv: str, size: int, overwrite: bool = False) -> None:
    import pandas as pd
    df = pd.read_csv(index_csv)
    out_paths, done, failed = [], 0, 0
    for i, src in enumerate(df["image_path"]):
        dst = cache_path_for(src, size)
        if overwrite or not dst.exists():
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                sq, _, _ = preprocess_fundus(src, size)
                cv2.imwrite(str(dst), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR))
                done += 1
            except Exception as e:  # keep going; report at the end
                print(f"  ! failed {src}: {e}")
                dst = None
                failed += 1
        out_paths.append(str(dst) if dst else "")
        if (i + 1) % 250 == 0:
            print(f"  cached {i + 1}/{len(df)}")
    df[f"cached_{size}"] = out_paths
    df.to_csv(index_csv, index=False)
    print(f"Done. New: {done}, failed: {failed}. Column 'cached_{size}' written to {index_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crop/resize fundus images and cache them.")
    ap.add_argument("--index", help="index CSV with an image_path column")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", help="run the quality check on a single image and exit")
    a = ap.parse_args()
    if a.check:
        import json
        print(json.dumps(quality_check(a.check), indent=2))
    elif a.index:
        cache_index(a.index, a.size, a.overwrite)
    else:
        ap.error("give --index <csv> to cache a dataset, or --check <image> for a quality check")
