"""
EyeVision AI web application (FastAPI).

    python run_app.py              # then open http://127.0.0.1:8000
    (or) uvicorn app.main:app --port 8000

Research/educational prototype - NOT a medical device.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import config  # noqa: E402
from app.db import Database  # noqa: E402
from config import DISEASE_CODES, DISEASE_NAMES, SHORT_NAMES  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 20 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
STATE = {}


@asynccontextmanager
async def lifespan(app):
    from inference import EyeVisionEngine
    STATE["db"] = Database(config.DB_PATH)
    STATE["engine"] = EyeVisionEngine()
    yield


app = FastAPI(title="EyeVision AI (research prototype)", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/uploads", StaticFiles(directory=config.UPLOAD_DIR), name="uploads")
app.mount("/heatmaps", StaticFiles(directory=config.HEATMAP_DIR), name="heatmaps")
app.mount("/reports", StaticFiles(directory=config.REPORTS_DIR), name="reports")


def db() -> Database:
    return STATE["db"]


def rel(path) -> str:
    """Store paths relative to the project folder so the database stays portable."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def absolute(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def public_url(path: str | None) -> str | None:
    if not path:
        return None
    p = absolute(path)
    for base, prefix in ((config.UPLOAD_DIR, "/uploads/"), (config.HEATMAP_DIR, "/heatmaps/")):
        try:
            return prefix + str(p.resolve().relative_to(base.resolve())).replace("\\", "/")
        except ValueError:
            continue
    return None


def present(exam: dict) -> dict:
    e = dict(exam)
    e["image_url"] = public_url(e.pop("image_path", None))
    e["heatmap_url"] = public_url(e.pop("heatmap_path", None))
    return e


# --------------------------------------------------------------------------- pages
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta():
    return {"labels": [{"code": c, "name": DISEASE_NAMES[c], "short": SHORT_NAMES[c]} for c in DISEASE_CODES],
            "disclaimer": config.DISCLAIMER, "se_note": config.SE_NOTE, "gradcam_note": config.GRADCAM_NOTE,
            "trend_note": config.TREND_NOTE, "models": STATE["engine"].status()}


# --------------------------------------------------------------------------- dashboard
@app.get("/api/dashboard")
def dashboard():
    d = db()
    latest = d.latest_per_patient_eye()
    risk = []
    for c in DISEASE_CODES:
        if c == "N":
            continue
        probs = [e["disease_probs"][c] for e in latest if e.get("disease_probs")]
        above = sum(1 for e in latest if e.get("disease_probs") and e.get("disease_thresholds")
                    and e["disease_probs"][c] >= e["disease_thresholds"][c])
        risk.append({"code": c, "name": DISEASE_NAMES[c], "short": SHORT_NAMES[c],
                     "eyes_above_threshold": above, "eyes": len(probs),
                     "mean_probability": (sum(probs) / len(probs)) if probs else None})
    recent = []
    for v in d.recent_visits(8):
        ex = d.exams_for_visit(v["visit_id"])
        top = None
        for e in ex:
            if e.get("disease_probs"):
                cand = max(((c, p) for c, p in e["disease_probs"].items() if c != "N"), key=lambda t: t[1])
                if top is None or cand[1] > top[1]:
                    top = cand
        recent.append({**v, "eyes": [e["eye"] for e in ex],
                       "se": {e["eye"]: e["predicted_se"] for e in ex if e["predicted_se"] is not None},
                       "top_class": ({"code": top[0], "short": SHORT_NAMES[top[0]], "probability": top[1]}
                                     if top else None),
                       "demo": any(e["disease_is_demo"] or e["refractive_is_demo"] for e in ex)})
    latest_se = sorted([{"patient_id": e["patient_id"], "eye": e["eye"], "exam_date": e["exam_date"],
                         "predicted_se": e["predicted_se"], "demo": bool(e["refractive_is_demo"])}
                        for e in latest if e["predicted_se"] is not None],
                       key=lambda r: r["exam_date"], reverse=True)[:10]
    return {"counts": d.counts(), "risk_overview": risk, "recent_visits": recent, "latest_se": latest_se}


# --------------------------------------------------------------------------- patients
@app.get("/api/patients")
def patients():
    return db().list_patients()


@app.get("/api/patients/{patient_id}")
def patient(patient_id: str):
    p = db().get_patient(patient_id)
    if not p:
        raise HTTPException(404, f"No patient with ID '{patient_id}'")
    visits = {}
    for e in db().exams_for_patient(patient_id):
        v = visits.setdefault(e["visit_id"], {"visit_id": e["visit_id"], "exam_date": e["exam_date"], "eyes": {}})
        v["eyes"][e["eye"]] = present(e)
    return {"patient": p, "visits": sorted(visits.values(), key=lambda v: v["exam_date"], reverse=True)}


@app.get("/api/patients/{patient_id}/trends")
def trends(patient_id: str):
    exams = db().exams_for_patient(patient_id)
    if not exams:
        raise HTTPException(404, f"No examinations for '{patient_id}'")
    out = {"patient_id": patient_id, "note": config.TREND_NOTE, "eyes": {}}
    for eye in ("R", "L"):
        ex = [e for e in exams if e["eye"] == eye]
        if not ex:
            continue
        series = [{"exam_date": e["exam_date"], "examination_id": e["examination_id"],
                   "se": e["predicted_se"], "se_low": e["se_interval_low"], "se_high": e["se_interval_high"],
                   "se_sd": e["se_uncertainty"], "probs": e["disease_probs"],
                   "demo": bool(e["disease_is_demo"] or e["refractive_is_demo"]),
                   "model": e["refractive_model_version"]} for e in ex]
        change = None
        se_pts = [s for s in series if s["se"] is not None]
        if len(se_pts) >= 2:
            a, b = se_pts[-2], se_pts[-1]
            delta = b["se"] - a["se"]
            combined = ((a["se_sd"] or 0) ** 2 + (b["se_sd"] or 0) ** 2) ** 0.5
            same_model = a["model"] == b["model"]
            change = {"from_date": a["exam_date"], "to_date": b["exam_date"], "delta_se": delta,
                      "delta_since_first": b["se"] - se_pts[0]["se"], "first_date": se_pts[0]["exam_date"],
                      "combined_uncertainty_sd": combined,
                      "exceeds_uncertainty": abs(delta) > 1.96 * combined if combined else None,
                      "same_model_version": same_model,
                      "direction": "more negative (more myopic estimate)" if delta < 0
                      else "more positive" if delta > 0 else "unchanged"}
        prob_change = None
        pr = [s for s in series if s["probs"]]
        if len(pr) >= 2:
            prob_change = {c: pr[-1]["probs"][c] - pr[-2]["probs"][c] for c in DISEASE_CODES}
        out["eyes"][eye] = {"series": series, "se_change": change, "prob_change": prob_change}
    return out


# --------------------------------------------------------------------------- examinations
def _validate(patient_id, age, sex, exam_date):
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", patient_id or ""):
        raise HTTPException(422, "Patient ID: use 1-40 letters, digits, '-' or '_'.")
    if age is not None and not (0 <= age <= 120):
        raise HTTPException(422, "Age must be between 0 and 120.")
    if sex not in ("M", "F", "O", "U"):
        raise HTTPException(422, "Sex must be M, F, O or U.")
    try:
        d = dt.date.fromisoformat(exam_date)
    except ValueError:
        raise HTTPException(422, "Examination date must be YYYY-MM-DD.")
    if d > dt.date.today() + dt.timedelta(days=1):
        raise HTTPException(422, "Examination date is in the future.")


async def _save_upload(f: UploadFile, visit_id: str, eye: str) -> Path:
    ext = Path(f.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(422, f"{'Left' if eye == 'L' else 'Right'} eye: unsupported file type '{ext}'. "
                                 f"Use JPG, PNG, TIFF or BMP.")
    data = await f.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 20 MB.")
    if not data:
        raise HTTPException(422, "Empty file.")
    path = config.UPLOAD_DIR / f"{visit_id}_{eye}{ext}"
    path.write_bytes(data)
    return path


def run_analysis(visit_id, patient_id, age, sex, exam_date, images: dict, engine=None):
    """images: {'L': path, 'R': path}. Stores one examination row per eye."""
    engine = engine or STATE["engine"]
    db().upsert_patient(patient_id, age, sex)
    results = {}
    for eye, path in images.items():
        heat = config.HEATMAP_DIR / f"{visit_id}_{eye}.png"
        try:
            r = engine.analyze(str(path), gradcam_out=str(heat))
        except ValueError as e:
            raise HTTPException(422, f"{'Left' if eye == 'L' else 'Right'} eye image could not be read: {e}")
        dis, ref = r.get("disease"), r.get("refractive")
        rec = {
            "visit_id": visit_id, "patient_id": patient_id, "exam_date": exam_date, "eye": eye,
            "age_at_exam": age, "image_path": rel(path),
            "heatmap_path": rel(heat) if r.get("gradcam", {}) and r["gradcam"].get("path") else None,
            "heatmap_class": (r.get("gradcam") or {}).get("class"),
            "disease_probs": {c["code"]: c["probability"] for c in dis["classes"]} if dis else None,
            "disease_uncertainty": {c["code"]: c["uncertainty"] for c in dis["classes"]} if dis else None,
            "disease_thresholds": {c["code"]: c["threshold"] for c in dis["classes"]} if dis else None,
            "disease_confidence": dis["confidence"] if dis else None,
            "predicted_se": ref["estimated_se_raw"] if ref else None,
            "se_uncertainty": ref["uncertainty_sd"] if ref else None,
            "se_interval_low": ref["interval_95"][0] if ref else None,
            "se_interval_high": ref["interval_95"][1] if ref else None,
            "se_confidence": ref["confidence"] if ref else None,
            "quality": r["quality"],
            "disease_model_version": dis["model_version"] if dis else None,
            "refractive_model_version": ref["model_version"] if ref else None,
            "disease_is_demo": int(bool(dis and dis["is_demo_model"])),
            "refractive_is_demo": int(bool(ref and ref["is_demo_model"])),
        }
        db().add_exam(rec)
        results[eye] = r
    return results


@app.post("/api/examinations")
async def create_examination(patient_id: str = Form(...), age: int | None = Form(None), sex: str = Form("U"),
                             exam_date: str = Form(...), left_image: UploadFile | None = File(None),
                             right_image: UploadFile | None = File(None)):
    patient_id = patient_id.strip()
    sex = (sex or "U").upper()[:1]
    _validate(patient_id, age, sex, exam_date)
    files = {k: v for k, v in (("L", left_image), ("R", right_image)) if v is not None and v.filename}
    if not files:
        raise HTTPException(422, "Upload at least one eye image.")
    visit_id = uuid.uuid4().hex[:12]
    images = {eye: await _save_upload(f, visit_id, eye) for eye, f in files.items()}
    from starlette.concurrency import run_in_threadpool
    await run_in_threadpool(run_analysis, visit_id, patient_id, age, sex, exam_date, images)
    return visit(visit_id)


@app.get("/api/visits/{visit_id}")
def visit(visit_id: str):
    ex = db().exams_for_visit(visit_id)
    if not ex:
        raise HTTPException(404, "Examination not found")
    p = db().get_patient(ex[0]["patient_id"])
    return {"visit_id": visit_id, "patient": p, "exam_date": ex[0]["exam_date"],
            "eyes": {e["eye"]: present(e) for e in ex}, "disclaimer": config.DISCLAIMER}


@app.delete("/api/visits/{visit_id}")
def delete_visit(visit_id: str):
    rows = db().delete_visit(visit_id)
    if not rows:
        raise HTTPException(404, "Examination not found")
    for r in rows:
        for k in ("image_path", "heatmap_path"):
            if r.get(k):
                absolute(r[k]).unlink(missing_ok=True)
    return {"deleted": visit_id}


# --------------------------------------------------------------------------- model info
@app.get("/api/model-info")
def model_info():
    st = STATE["engine"].status()
    for task in ("disease", "refractive"):
        m = st.get(task)
        if not m:
            continue
        stem = Path(m["file"]).stem
        rep = config.REPORTS_DIR / stem
        m["report_pdf"] = f"/reports/{stem}/evaluation_report.pdf" if (rep / "evaluation_report.pdf").exists() else None
        m["plots"] = [f"/reports/{stem}/{n}" for n in ("roc.png", "pr.png", "confusion.png", "regression.png")
                      if (rep / n).exists()]
        hist = config.REPORTS_DIR / f"{stem}_history.png"
        m["history_plot"] = f"/reports/{hist.name}" if hist.exists() else None
    comp = config.REPORTS_DIR / "comparison_disease.csv"
    if comp.exists():
        import csv
        st["comparison"] = list(csv.DictReader(comp.open()))
    return st
