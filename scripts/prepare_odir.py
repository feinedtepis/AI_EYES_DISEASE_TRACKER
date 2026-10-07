"""
Build the per-eye ODIR-5K index (data/odir_index.csv) and cache resized images.

    python scripts/prepare_odir.py --root path/to/ODIR-5K-folder [--size 224]

Works with the official release (ODIR-5K_Training_Annotations(Updated)_V2.xlsx +
ODIR-5K_Training_Dataset/) and with the Kaggle release (data.xlsx / full_df.csv +
preprocessed_images/ or 'Training Images/').
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from dataset import build_odir_index, index_path  # noqa: E402
from preprocess import cache_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--size", type=int, default=0, help="also cache images at this size (e.g. 224 or 384)")
a = ap.parse_args()
df = build_odir_index(a.root)
out = index_path("odir")
df.to_csv(out, index=False)
print(f"Saved {out}")
print("Per-eye label counts:", {c: int(df[c].sum()) for c in config.DISEASE_CODES})
if a.size:
    cache_index(str(out), a.size)
