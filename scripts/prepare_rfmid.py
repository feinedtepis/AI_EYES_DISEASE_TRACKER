"""
Build the RFMiD index (data/rfmid_index.csv) for external evaluation.

    python scripts/prepare_rfmid.py --root path/to/RFMiD [--size 224]
    python evaluate.py --task disease --external rfmid
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from dataset import build_rfmid_index, index_path  # noqa: E402
from preprocess import cache_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--size", type=int, default=0)
a = ap.parse_args()
df = build_rfmid_index(a.root)
out = index_path("rfmid")
df.to_csv(out, index=False)
print(f"Saved {out}")
print("Mapped label counts:", {c: int(df[c].sum()) for c in config.DISEASE_CODES if df[c].notna().any()})
if a.size:
    cache_index(str(out), a.size)
