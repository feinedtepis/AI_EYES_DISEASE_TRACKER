"""
Single-image prediction from the command line.

    python predict.py --image path/to/fundus.jpg
    python predict.py --image left.jpg --gradcam reports/left_cam.png --json

Uses the best trained models in models/ (falls back to the synthetic DEMO
models in models/demo/, which are NOT medically meaningful).
"""
from __future__ import annotations

import argparse
import json

import config
from inference import EyeVisionEngine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--gradcam", default="", help="save a Grad-CAM side-by-side PNG here")
    ap.add_argument("--mc", type=int, default=config.MC_DROPOUT_PASSES, help="MC-dropout passes (0/1 = off)")
    ap.add_argument("--disease-model", default=None)
    ap.add_argument("--refractive-model", default=None)
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    eng = EyeVisionEngine(a.device, disease_ckpt=a.disease_model, refractive_ckpt=a.refractive_model)
    r = eng.analyze(a.image, gradcam_out=a.gradcam or None, mc_passes=a.mc)
    if a.json:
        print(json.dumps(r, indent=2, default=float))
        return

    line = "=" * 64
    print(line)
    print("EyeVision AI - experimental research estimate")
    print(line)
    q = r["quality"]
    print(f"Image quality check: {'OK' if q['gradable'] else 'POOR'}")
    for w in q["warnings"]:
        print(f"  ! {w}")
    if r.get("quality_warning"):
        print(f"  ! {r['quality_warning']}")
    if r["disease"]:
        d = r["disease"]
        print("\nDisease pattern analysis" + ("  [DEMO MODEL - synthetic data, meaningless]" if d["is_demo_model"] else ""))
        for c in sorted(d["classes"], key=lambda c: -c["probability"]):
            mark = "  <- above model threshold" if c["above_threshold"] and c["code"] != "N" else ""
            print(f"  {c['name']:<26} {100 * c['probability']:5.1f}%  (±{100 * c['uncertainty']:.1f}){mark}")
        print(f"  Model confidence (decisiveness): {100 * d['confidence']:.0f}%")
        print(f"  {d['summary']}")
    else:
        print("\nNo disease model found in models/ - run train.py --task disease first.")
    if r["refractive"]:
        s = r["refractive"]
        print("\nVision estimate" + ("  [DEMO MODEL - synthetic data, meaningless]" if s["is_demo_model"] else ""))
        print(f"  Estimated spherical equivalent: {s['estimated_se']:+.2f} D")
        print(f"  Approx. 95% interval: {s['interval_95'][0]:+.2f} to {s['interval_95'][1]:+.2f} D")
        print(f"  Model confidence: {100 * s['confidence']:.0f}%  (P(|error| <= {config.SE_CONFIDENCE_TOLERANCE_D:.2f} D))")
        print(f"  {s['note']}")
    else:
        print("\nNo refractive model found - SE estimate not available (needs a refraction-labelled dataset).")
    if r.get("gradcam") and r["gradcam"].get("path"):
        print(f"\nGrad-CAM saved: {r['gradcam']['path']}  ({r['gradcam']['class']})")
        print(f"  {config.GRADCAM_NOTE}")
    print("\n" + config.DISCLAIMER)
    print(line)


if __name__ == "__main__":
    main()
