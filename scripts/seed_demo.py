"""
Fill the database with two clearly-labelled DEMO patients so the History and
Trends pages have something to show before real follow-up data exists.

The images are SYNTHETIC cartoons (scripts/make_synthetic_data.py) analysed by the
synthetic DEMO models in models/demo/. Every stored record is flagged as demo.

    python scripts/seed_demo.py            # adds DEMO-001 (myopia progression) and DEMO-002 (DR lesions appear)
    python scripts/seed_demo.py --reset    # removes existing DEMO-* records first
"""
import argparse
import sys
import uuid
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config  # noqa: E402
from app import main as webapp  # noqa: E402
from app.db import Database  # noqa: E402
from inference import EyeVisionEngine  # noqa: E402
from make_synthetic_data import render_fundus, save  # noqa: E402

PLAN = {
    "DEMO-001": {"age": 14, "sex": "F", "visits": [
        ("2025-03-10", {}, -1.25), ("2025-09-14", {}, -1.75), ("2026-03-12", {}, -2.25), ("2026-09-20", {}, -2.75)]},
    "DEMO-002": {"age": 58, "sex": "M", "visits": [
        ("2025-08-02", {}, -0.50), ("2026-02-05", {"D": 1}, -0.50), ("2026-08-09", {"D": 1, "H": 1}, -0.75)]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    dis = next(iter(sorted(config.DEMO_MODELS_DIR.glob("disease_*.pth"))), None)
    ref = next(iter(sorted(config.DEMO_MODELS_DIR.glob("refractive_*.pth"))), None)
    engine = EyeVisionEngine(disease_ckpt=dis, refractive_ckpt=ref)
    engine.disease_demo = True
    engine.refr_demo = True
    db = Database(config.DB_PATH)
    webapp.STATE["db"] = db
    if a.reset:
        with db.conn() as c:
            for (vid,) in c.execute("SELECT DISTINCT visit_id FROM examinations WHERE patient_id LIKE 'DEMO-%'").fetchall():
                for r in db.delete_visit(vid):
                    for k in ("image_path", "heatmap_path"):
                        if r.get(k):
                            webapp.absolute(r[k]).unlink(missing_ok=True)
            c.execute("DELETE FROM patients WHERE patient_id LIKE 'DEMO-%'")
    rng = np.random.default_rng(11)
    for pid, p in PLAN.items():
        pig = rng.uniform(0.85, 1.05)
        for i, (date, labels, se) in enumerate(p["visits"]):
            vid = uuid.uuid4().hex[:12]
            images = {}
            for eye, off in (("R", 0.0), ("L", 0.25)):
                img = render_fundus(np.random.default_rng(zlib.crc32(f'{pid}{eye}{i}'.encode())), eye, labels, se + off, p["age"], pig)
                path = config.UPLOAD_DIR / f"{vid}_{eye}.png"
                save(img, path)
                images[eye] = path
            webapp.run_analysis(vid, pid, p["age"] + i // 2, p["sex"], date, images, engine=engine)
            print(f"  {pid} {date} stored")
    print("Demo patients added (flagged as demo).")


if __name__ == "__main__":
    main()
