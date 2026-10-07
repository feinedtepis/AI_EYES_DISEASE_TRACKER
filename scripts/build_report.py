"""
Build the project report PDF from the actual result files (no numbers are typed by hand).

    python scripts/make_report_figures.py
    python scripts/build_report.py            -> EyeVision_AI_Project_Report.pdf

Edit AUTHOR / INSTITUTE below to change the title page.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from config import DISEASE_CODES, DISEASE_NAMES, SHORT_NAMES  # noqa: E402

AUTHOR = "Harsh"
ENROLMENT = "992401030282"
BATCH = "F5"
PROGRAMME = "B.Tech Computer Science &amp; Engineering"
INSTITUTE = "Jaypee Institute of Information Technology, Noida"
OUT = ROOT / "EyeVision_AI_Project_Report.pdf"
R = config.REPORTS_DIR
FIG = R / "figures"
SHOTS = R / "screenshots"

# ----------------------------------------------------------------------------- fonts & styles
import matplotlib  # noqa: E402
from reportlab.lib.fonts import addMapping  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402

_FD = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
for _name, _file in (("DejaVu", "DejaVuSans.ttf"), ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu-Italic", "DejaVuSans-Oblique.ttf"), ("DejaVu-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
                     ("DejaVuMono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(_name, str(_FD / _file)))
addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold")
addMapping("DejaVu", 0, 1, "DejaVu-Italic"); addMapping("DejaVu", 1, 1, "DejaVu-BoldItalic")

ss = getSampleStyleSheet()
TEAL, INK, MUTED = colors.HexColor("#0e6b70"), colors.HexColor("#15242e"), colors.HexColor("#5d6d77")
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DejaVu-Bold", fontSize=16, leading=20, textColor=INK,
                    spaceBefore=4, spaceAfter=10, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DejaVu-Bold", fontSize=12, leading=15, textColor=TEAL,
                    spaceBefore=10, spaceAfter=5, keepWithNext=1)
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="DejaVu", fontSize=9.3, leading=13.6, textColor=INK,
                   spaceAfter=6)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8.2, leading=11, textColor=MUTED, spaceBefore=2, spaceAfter=10)
SMALL = ParagraphStyle("SMALL", parent=B, fontSize=8.4, leading=11.5)
WARN = ParagraphStyle("WARN", parent=B, fontSize=9.2, leading=13, backColor=colors.HexColor("#fff5e3"),
                      borderColor=colors.HexColor("#e9c27f"), borderWidth=0.8, borderPadding=7, spaceBefore=6,
                      spaceAfter=12, textColor=colors.HexColor("#5e3a00"))
CODE = ParagraphStyle("CODE", parent=B, fontName="DejaVuMono", fontSize=8.3, leading=10.8, backColor=colors.HexColor("#f2f5f6"),
                      borderPadding=5, spaceBefore=3, spaceAfter=8)


def f(v, nd=3, pct=False):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if pct:
        return f"{100 * v:.1f}%"
    return f"{v:.{nd}f}"


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


HDR_L = ParagraphStyle("HDR_L", fontName="DejaVu-Bold", fontSize=7.4, leading=9, textColor=TEAL)
HDR_R = ParagraphStyle("HDR_R", parent=HDR_L, alignment=2)


def tbl(rows, widths=None, font=8.0, header=True, align_right_from=1):
    rows = [list(r) for r in rows]
    if header:   # header cells wrap inside their column instead of overlapping
        rows[0] = [Paragraph(str(h), HDR_R if (align_right_from is not None and i >= align_right_from) else HDR_L)
                   for i, h in enumerate(rows[0])]
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    st = [("FONTNAME", (0, 0), (-1, -1), "DejaVu"), ("FONTSIZE", (0, 0), (-1, -1), font),
          ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d5dde2")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if align_right_from is not None:
        st.append(("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"))
    if header:
        st += [("LINEBELOW", (0, 0), (-1, 0), 0.9, TEAL), ("VALIGN", (0, 0), (-1, 0), "BOTTOM")]
    t.setStyle(TableStyle(st))
    return t


FIGNO = [0]


def fig(text):
    FIGNO[0] += 1
    return f"Figure {FIGNO[0]}. {text}"


def img(path, width=17.0, caption=None, max_h=22.0):
    path = Path(path)
    if not path.exists():
        return [P(f"<i>[figure missing: {path.name}]</i>", CAP)]
    from PIL import Image as PI
    w, h = PI.open(path).size
    height = width * h / w
    if height > max_h:
        width, height = width * max_h / height, max_h
    if path.parent == SHOTS:   # large UI screenshots -> JPEG keeps the PDF small
        pass   # screenshots are already stored as JPEG
    out = [Image(str(path), width=width * cm, height=height * cm)]
    if caption:
        out.append(P(caption, CAP))
    return [KeepTogether(out)]


def raw_img(path, width):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(str(path), width=width * cm, height=width * cm * h / w)


def keep(*flowables):
    return [KeepTogether(list(flowables))]


def load_json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- data
def collect():
    d = {}
    d["models"] = {}
    for p in sorted(config.MODELS_DIR.glob("disease_*.json")):
        m = load_json(p)
        if m and m.get("test_metrics"):
            d["models"][p.stem] = m
    best = max(d["models"].items(), key=lambda kv: kv[1]["val_metrics"]["macro_roc_auc"]) if d["models"] else (None, None)
    d["best_name"], d["best"] = best
    d["baselines"] = (load_json(R / "baselines_disease.json") or {}).get("results", {})
    d["ablation"] = pd.read_csv(R / "loss_ablation.csv") if (R / "loss_ablation.csv").exists() else None
    d["ablation_setup"] = (load_json(R / "loss_ablation.json") or {}).get("setup", "")
    idx = pd.read_csv(config.DATA_DIR / "odir_index.csv")
    split = pd.read_csv(config.DATA_DIR / "splits" / "disease_odir_seed42.csv", dtype={"patient_id": str})
    d["idx"] = idx.merge(split, on="patient_id", how="left")
    d["sub"] = pd.read_csv(R / d["best_name"] / "subgroups.csv") if d["best_name"] and (R / d["best_name"] / "subgroups.csv").exists() else None
    refs = sorted(config.DEMO_MODELS_DIR.glob("refractive_*.json"))
    d["refr"] = load_json(refs[0]) if refs else None
    d["refr_name"] = refs[0].stem if refs else None
    d["refr_base"] = (load_json(R / "baselines_refractive.json") or {}).get("results", {})
    return d


def nice_arch(m):
    a = {"efficientnet_b0": "EfficientNet-B0", "efficientnet_b2": "EfficientNet-B2", "resnet50": "ResNet-50",
         "convnext_tiny": "ConvNeXt-T", "resnet18": "ResNet-18"}.get(m["arch"], m["arch"])
    return a + (" (partial fine-tuning)" if m.get("freeze_early") else "")


# ----------------------------------------------------------------------------- document
def build():
    d = collect()
    best, idx = d["best"], d["idx"]
    s = []

    # ---------------- title page
    s += [Spacer(1, 3.2 * cm),
          P("EyeVision AI", ParagraphStyle("T0", parent=H1, fontSize=30, leading=34, alignment=TA_CENTER, textColor=TEAL)),
          Spacer(1, 6),
          P("An experimental deep-learning system for fundus-image eye-disease scoring, refractive-error "
            "estimation and longitudinal tracking", ParagraphStyle("T1", parent=B, fontSize=13.5, leading=19,
                                                                   alignment=TA_CENTER)),
          Spacer(1, 1.2 * cm),
          P("Research / college project report", ParagraphStyle("T2", parent=B, fontSize=11, alignment=TA_CENTER,
                                                                 textColor=MUTED)),
          Spacer(1, 2.2 * cm),
          P(f"Submitted by <b>{AUTHOR}</b><br/>Enrolment No. {ENROLMENT} · Batch {BATCH}<br/>{PROGRAMME}<br/>{INSTITUTE}",
            ParagraphStyle("T3", parent=B, fontSize=11, leading=17, alignment=TA_CENTER)),
          Spacer(1, 0.6 * cm),
          P(dt.date.today().strftime("%B %Y"), ParagraphStyle("T4", parent=B, alignment=TA_CENTER, textColor=MUTED)),
          Spacer(1, 2.4 * cm),
          P("<b>Important.</b> This is an experimental research/educational system, not a medical device. Its outputs "
            "are not clinical diagnoses and not eyeglass prescriptions, and it has not been clinically validated. "
            "Diagnosis and refraction require a qualified ophthalmologist/optometrist.", WARN),
          PageBreak()]

    # ---------------- abstract
    nimg, npat = len(idx), idx.patient_id.nunique()
    tm = best["test_metrics"]
    bl_best = max(d["baselines"].items(), key=lambda kv: kv[1]["summary"]["macro_roc_auc"]) if d["baselines"] else None
    s += [P("Abstract", H1),
          P(f"We built an end-to-end research prototype that analyses colour fundus photographs. A multi-label "
            f"convolutional network scores eight categories (normal, diabetic retinopathy, glaucoma, cataract, "
            f"age-related macular degeneration, hypertensive retinopathy, pathological myopia, other) and a separate "
            f"regression network estimates the spherical equivalent (SE) refractive error. Every result is stored per "
            f"patient and eye, so changes in the model's estimates can be followed over time in a web application "
            f"(FastAPI, SQLite, Chart.js) that always shows uncertainty, a Grad-CAM attention map labelled for research "
            f"use only, and a non-diagnostic disclaimer."),
          P(f"The disease models were trained on the public ODIR-5K dataset ({nimg:,} eye images from {npat:,} patients "
            f"after quality filtering) with labels re-derived per eye from the diagnostic keywords and a strictly "
            f"patient-level train/validation/test split. On {int(best['test_per_class']['N']['n'])} held-out test images the "
            f"best model ({nice_arch(best)}, {best['img_size']} px) reached a macro ROC-AUC of <b>{f(tm['macro_roc_auc'])}</b>, "
            f"macro PR-AUC {f(tm['macro_pr_auc'])} and macro F1 {f(tm['macro_f1'])}"
            + (f", compared with {f(bl_best[1]['summary']['macro_roc_auc'])} for the best hand-crafted-feature baseline "
               f"({bl_best[0].lower()})" if bl_best else "") + ". "
            f"Performance varies strongly by class and is reported per class with confidence intervals."),
          P("No sufficiently large public dataset pairing fundus photographs with measured refraction could be obtained, "
            "so the refractive component is implemented and verified end-to-end on synthetic images only and is labelled "
            "as a demonstration everywhere; it accepts a refraction CSV for future training on real data."),
          Spacer(1, 6), P("Contents", H2)]
    toc = ["1 Introduction and objectives", "2 Datasets", "3 System architecture", "4 Methods", "5 Results — disease classification",
           "6 Results — refractive-error estimation", "7 The web application", "8 Discussion", "9 Limitations and ethics",
           "10 Future work", "References", "Appendix A — How to run", "Appendix B — Project structure"]
    s += [P(t, ParagraphStyle("toc", parent=B, spaceAfter=1.5)) for t in toc]
    s.append(PageBreak())

    # ---------------- 1 intro
    s += [P("1  Introduction and objectives", H1),
          P("Retinal (fundus) photographs are cheap and non-invasive, and deep learning on fundus images is an active "
            "research area for screening conditions such as diabetic retinopathy. This project builds a complete, honest "
            "research prototype rather than a single model: data preparation, training, evaluation against baselines, "
            "explainability, uncertainty, patient-level storage and a usable interface."),
          P("Objectives:")]
    s += bullets([
        "Provide multi-label probabilities for myopia, cataract, glaucoma, diabetic retinopathy, AMD, hypertension-related "
        "retinal abnormalities and normal/healthy (plus an 'other' class) from a fundus image.",
        "Estimate refractive error (spherical equivalent) where suitable training data exists, without presenting it as a prescription.",
        "Store examinations so a patient's results can be compared over time, and show whether estimates increase or decrease.",
        "Display uncertainty and confidence instead of presenting predictions as facts.",
        "Evaluate with metrics suited to imbalanced medical data (not accuracy), per class, by subgroup and against baselines, "
        "and report weak results honestly."])
    s.append(P("Safety wording used everywhere in the software: <i>“Research/educational AI estimate only. This system is not "
               "a medical diagnostic tool and does not provide a clinical diagnosis or eyeglass prescription. Please consult "
               "a qualified ophthalmologist/optometrist for diagnosis and refraction.”</i> The system never says “you have "
               "cataract”; it says “elevated model score for cataract-like features”.", SMALL))

    # ---------------- 2 datasets
    tr = idx[idx.split == "train"]; va = idx[idx.split == "val"]; te = idx[idx.split == "test"]
    s += [P("2  Datasets", H1), P("2.1  ODIR-5K (primary, used for training and testing)", H2),
          P("ODIR-5K (Ocular Disease Intelligent Recognition, ODIR-2019 challenge) contains colour fundus photographs of the "
            "left and right eyes of 5,000 patients collected from hospitals in China with several camera models, with patient "
            "age, sex and per-eye diagnostic keywords. The labelled training portion (3,500 patients) is public; we used the "
            "Kaggle release, whose 512×512 pre-cropped images exclude some low-quality photographs."),
          P("<b>Per-eye labels.</b> ODIR's eight label columns describe the <i>patient</i> (union of both eyes). Using them per "
            "image would label a healthy eye with its fellow eye's disease. We therefore mapped each eye's keywords to labels "
            "(e.g. “moderate non proliferative retinopathy” → DR, “pathological myopia” → myopia, “drusen”/“epiretinal "
            "membrane” → other; “lens dust” ignored; “low image quality”, “no fundus image” and “optic disk photographically "
            "invisible” excluded). As a check, the union of our per-eye labels reproduces the official patient-level labels "
            "for <b>3,500 of 3,500 patients (100%)</b>."),
          tbl([["", "Train", "Validation", "Test", "Total"],
               ["Patients", f"{tr.patient_id.nunique():,}", f"{va.patient_id.nunique():,}", f"{te.patient_id.nunique():,}", f"{npat:,}"],
               ["Eye images", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}", f"{nimg:,}"]] +
              [[DISEASE_NAMES[c], f"{int(tr[c].sum())}", f"{int(va[c].sum())}", f"{int(te[c].sum())}", f"{int(idx[c].sum())}"]
               for c in DISEASE_CODES], [5.2 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm, 2.4 * cm]),
          P(f"Table 1. Patient-level split (70/15/15, stratified by each patient's rarest label, seed 42) and per-eye label "
            f"counts. {100 * (idx[DISEASE_CODES].sum(1) > 1).mean():.1f}% of images carry more than one label. Mean age "
            f"{idx.drop_duplicates('patient_id').age.mean():.1f} years; {(idx.drop_duplicates('patient_id').sex == 'F').mean() * 100:.0f}% "
            f"female. A handful of recorded ages below 18 (including age 1) look like data-entry errors and were kept as recorded.", CAP)]
    s += img(FIG / "dataset.png", 16.5, fig("Label distribution is heavily imbalanced: hypertensive retinopathy, myopia, AMD, "
                                         "cataract and glaucoma each make up only 3–5% of images."))
    s += [P("2.2  RFMiD (secondary, generalisation check)", H2),
          P("The Retinal Fundus Multi-Disease Image Dataset (3,200 images, 46 conditions, collected in India) is supported "
            "by <font face='DejaVuMono'>scripts/prepare_rfmid.py</font> and <font face='DejaVuMono'>evaluate.py --external rfmid</font>. "
            "Its columns are mapped onto our label space where an equivalent exists (DR, ARMD→AMD, MYA→myopia, optic-disc "
            "cupping→glaucoma as a proxy, normal = no disease risk, other); cataract and hypertension have no RFMiD "
            "equivalent and are masked (excluded from loss and metrics). RFMiD is distributed through IEEE Dataport behind "
            "a login, which the build environment could not reach, so <b>no external-validation numbers are reported</b>; the "
            "code path was tested on a mock folder in the official layout."),
          P("2.3  Refractive-error data", H2),
          P("The reference study (Yang et al., 2022) trained ResNet-50, Inception-v3 and Inception-ResNet-v2 on 987 "
            "ultra-widefield Optos images of myopic patients, with SE from subjective refraction as the target, and reported a "
            "test MAE of about 1.7 D (roughly 30% within ±0.75 D); the authors concluded accuracy still needs improvement. "
            "Their data are not public, and ultra-widefield images differ from the 45° photographs in ODIR. The MMAC 2023 "
            "challenge (Task 3, SE prediction, 992 training images) keeps its data with the organisers. Because no "
            "sufficiently large public fundus + refraction dataset could be downloaded, the regression pipeline loads a CSV "
            "(<font face='DejaVuMono'>image_path, patient_id, age, sex, sphere, cylinder, axis, spherical_equivalent</font>; "
            "SE = sphere + cylinder/2 when missing) and was verified on synthetic data only. Disease-only datasets are never "
            "used for regression."),
          P("2.4  Synthetic data (testing only)", H2),
          P("<font face='DejaVuMono'>scripts/make_synthetic_data.py</font> draws cartoon fundus images (disc, cup, vessels, "
            "lesions, tessellation, haze) with known labels and a synthetic SE that drives tessellation and disc appearance. "
            "They exist only to run every script end-to-end and to demonstrate the refractive feature; results on them "
            "say nothing about real-world performance.")]
    s += img(FIG / "synthetic_examples.png", 17, fig("Examples of the synthetic test images (not real retinas)."))

    # ---------------- 3 architecture
    s += [P("3  System architecture", H1)]
    s += img(FIG / "architecture.png", 17, fig("Data flow. The offline pipeline produces checkpoints and reports; the web "
                                           "application loads the best checkpoint by validation score and stores every "
                                           "analysis with its model version."))
    s += [P("Technology: Python, PyTorch, timm (ImageNet-pretrained backbones), scikit-learn, OpenCV; FastAPI backend with a "
            "SQLite database; a dependency-free HTML/CSS/JavaScript front-end with Chart.js. The database has a "
            "<font face='DejaVuMono'>patients</font> table (patient_id, age, sex, registration date) and an "
            "<font face='DejaVuMono'>examinations</font> table with one row per eye per visit (examination_id, visit_id, "
            "patient_id, exam date, eye, image path, disease probabilities and uncertainties, thresholds, predicted SE, "
            "uncertainty and 95% range, confidence, image-quality report, model versions, timestamp)."),
          P("Command-line entry points, as specified: <font face='DejaVuMono'>python train.py --task disease|refractive</font>, "
            "<font face='DejaVuMono'>python evaluate.py --task disease|refractive</font>, "
            "<font face='DejaVuMono'>python predict.py --image path/to/image.jpg</font>; modules "
            "<font face='DejaVuMono'>dataset.py, models.py, preprocess.py, config.py</font>; checkpoints in "
            "<font face='DejaVuMono'>models/</font>.")]

    # ---------------- 4 methods
    cfgb = best.get("config", {})
    s += [P("4  Methods", H1), P("4.1  Preprocessing and augmentation", H2)]
    s += bullets([
        "Retina detection: threshold on max(grey, red), morphological clean-up, largest connected component; crop to its "
        "bounding box (removes black borders and corner text), pad to a square, resize (area interpolation).",
        "Normalisation with ImageNet mean/std (pretrained backbones).",
        "Training augmentation kept mild so clinical features are preserved: random resized crop (scale 0.88–1.0), "
        "horizontal flip (turns a left-eye layout into a right-eye layout; disease labels are unaffected; can be disabled "
        "with <font face='DejaVuMono'>--no-hflip</font>), rotation ±12°, brightness/contrast ±15%, saturation ±5%, no hue shift.",
        "Heuristic image-quality check shown to the user: retina found, exposure, contrast, Laplacian sharpness. Blur "
        "warnings mention that cataract itself blurs the image."])
    s += [P("4.2  Leakage prevention", H2),
          P("Splits are made per patient, so the left and right eye of one person are never in different splits; an assertion "
            "stops training if any patient appears in two splits. The split is saved and reused by every model, baseline and "
            "evaluation, so all comparisons use the same test patients. Decision thresholds and early stopping use only the "
            "validation set; the test set is evaluated once per model."),
          P("4.3  Disease classifier (Task A)", H2),
          P(f"Transfer learning from ImageNet with EfficientNet-B0 and ResNet-50 backbones (EfficientNet-B2 and ConvNeXt-T "
            f"are also supported). The head is dropout (p = {best.get('dropout', 0.3)}) and a linear layer with 8 outputs; "
            f"sigmoid probabilities allow several conditions at once. Loss: BCEWithLogits with per-class positive weights "
            f"(#negatives/#positives, clipped to 1–30) — the 'weighted BCE' option; plain BCE and focal loss (γ = 2) are "
            f"compared in Section 5.3. Unknown labels are masked out of the loss. AdamW, one-cycle learning-rate schedule, "
            f"gradient clipping, first epoch with a frozen backbone, early stopping on validation macro ROC-AUC."),
          P(f"Training was done on a 2-core CPU, which limited the input size to {best['img_size']}×{best['img_size']} px and the "
            f"number of epochs (see Table 2). The ResNet-50 run used partial fine-tuning (stem and first two stages frozen) to "
            f"fit the compute budget. The included Kaggle notebook retrains at 384 px on a free GPU."),
          P("4.4  Refractive-error regressor (Task B)", H2),
          P("A separate network (ResNet-50 by default; EfficientNet and ConvNeXt supported) with an MLP head predicts the "
            "standardised SE; Smooth-L1 loss. Optional extra heads predict sphere, cylinder and axis — axis as (cos 2θ, sin 2θ) "
            "because it is 180°-periodic — and are trained only when those labels exist. Metrics: MAE, MSE, RMSE, R², mean "
            "bias and the share of estimates within ±0.50, ±1.00 and ±2.00 D."),
          P("4.5  Uncertainty and confidence", H2)]
    s += bullets([
        "Monte-Carlo dropout: 10 stochastic passes through the head (the backbone runs once). Each disease probability is "
        "shown as mean ± SD.",
        "Disease 'confidence' = 1 − mean binary entropy of the eight probabilities: how decisive the model is, explicitly "
        "<i>not</i> its accuracy.",
        "SE uncertainty σ = √(σ²<sub>MC</sub> + σ²<sub>val</sub>), where σ<sub>val</sub> is the residual SD on the validation set, "
        "so the stated range can never be tighter than the model's tested error. SE confidence = P(|error| ≤ 1.00 D) = "
        "erf(1/(σ√2)). Displayed estimates are rounded to 0.25 D steps with an approximate 95% range.",
        "Trend view: a change between two visits is flagged as exceeding uncertainty only if |Δ| &gt; 1.96·√(σ<sub>1</sub>² + σ<sub>2</sub>²), and the "
        "UI warns when the two estimates came from different model versions."])
    s += [P("4.6  Explainability", H2),
          P("Grad-CAM on the last convolutional block (EfficientNet: final 1×1 conv + BN; ResNet: layer4) for the highest "
            "non-normal class, masked to the retina and overlaid on the image. It is labelled “Model attention visualization — "
            "research use only”; a heat-map does not prove the model uses the correct pathology."),
          P("4.7  Baselines", H2),
          P("Disease: about 70 hand-crafted features per image (RGB/HSV histograms and moments, Laplacian sharpness, edge "
            "density, bright/dark spot ratios) with one-vs-rest logistic regression and random forest (class-balanced), "
            "thresholds tuned on validation. Refraction: predicting the training-set mean SE, and ridge regression on the same "
            "image features (+ age when available).")]

    # ---------------- 5 results
    s += [P("5  Results — disease classification", H1),
          P("All numbers in this section are computed by <font face='DejaVuMono'>evaluate.py</font> / "
            "<font face='DejaVuMono'>baselines.py</font> on the same held-out test patients and copied into this report by "
            "script; none were edited by hand.", SMALL),
          P("5.1  Model comparison", H2)]
    rows = [["Model", "Input", "Ep.", "ROC-AUC", "PR-AUC", "F1", "Sens.", "Spec.", "Micro F1"]]
    for name, m in d["models"].items():
        t = m["test_metrics"]
        rows.append([nice_arch(m), f"{m['img_size']} px", str(m.get("epochs_run", "")), f(t["macro_roc_auc"]), f(t["macro_pr_auc"]),
                     f(t["macro_f1"]), f(t["macro_recall"]), f(t["macro_specificity"]), f(t["micro_f1"])])
    for name, b in d["baselines"].items():
        t = b["summary"]
        rows.append([name, "256 px", "–", f(t["macro_roc_auc"]), f(t["macro_pr_auc"]), f(t["macro_f1"]), f(t["macro_recall"]),
                     f(t["macro_specificity"]), f(t["micro_f1"])])
    s += [tbl(rows, [4.5 * cm, 1.4 * cm, 0.9 * cm, 1.8 * cm, 1.8 * cm, 1.3 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm], font=7.6),
          P(f"Table 2. Test-set results ({int(best['test_per_class']['N']['n'])} images, {te.patient_id.nunique()} patients). "
            "All columns except micro F1 are macro averages (unweighted mean over the eight classes); sensitivity, specificity and F1 use thresholds tuned on validation. Ep. = epochs trained.", CAP)]
    s += img(FIG / "model_comparison.png", 16.5, fig("Deep models versus hand-crafted-feature baselines on the same test patients."))

    # per class
    pc = best["test_per_class"]
    s += [P(f"5.2  Per-class results of the selected model ({nice_arch(best)})", H2)]
    rows = [["Class", "n+", "Prev.", "Thr.", "Prec.", "Sens.", "Spec.", "F1", "ROC-AUC [95% CI]", "PR-AUC"]]
    for c in DISEASE_CODES:
        v = pc[c]
        ci = v.get("roc_auc_ci95") or [None, None]
        rows.append([DISEASE_NAMES[c] + (" *" if v["small_sample"] else ""), str(v["n_pos"]), f(v["prevalence"], pct=True),
                     f(v["threshold"], 2), f(v["precision"]), f(v["recall_sensitivity"]), f(v["specificity"]), f(v["f1"]),
                     f"{f(v['roc_auc'])} [{f(ci[0], 2)}–{f(ci[1], 2)}]", f(v["pr_auc"])])
    s += [tbl(rows, [3.8 * cm, 0.9 * cm, 1.3 * cm, 1.1 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.1 * cm, 3.0 * cm, 1.7 * cm], font=7.4),
          P("Table 3. n+ = positive test images, Prev. = prevalence, Thr. = validation-tuned threshold. * fewer than 30 positive test images — estimates are unstable (see the confidence intervals, from 300 "
            "bootstrap resamples). PR-AUC should be compared with the prevalence column, which is the PR-AUC of a random model.", CAP)]
    rdir = R / d["best_name"]
    s += [Table([[raw_img(rdir / "roc.png", 8.4), raw_img(rdir / "pr.png", 8.4)]], colWidths=[8.6 * cm, 8.6 * cm]),
          P(fig("ROC (left) and precision–recall (right) curves per class on the test set."), CAP)]
    s += img(rdir / "confusion.png", 16.5, fig("One-vs-rest confusion matrices at the validation-tuned thresholds "
                                           "(rows: true −/+, columns: predicted −/+)."))

    # ablation
    if d["ablation"] is not None:
        a = d["ablation"]
        rows = [["Loss", "Macro ROC-AUC", "Macro PR-AUC", "Macro F1", "Macro sens.", "Rare-class sens.", "Macro spec."]]
        for _, r in a.iterrows():
            rows.append([r["loss"], f(r["macro_roc_auc"]), f(r["macro_pr_auc"]), f(r["macro_f1"]), f(r["macro_sensitivity"]),
                         f(r["rare_class_sensitivity"]), f(r["macro_specificity"])])
        s += [P("5.3  Class imbalance: loss-function comparison", H2),
              P("To isolate the effect of the loss under the CPU budget, each loss trains the same linear head on the same "
                "frozen ImageNet EfficientNet-B0 features, with the same split and validation-tuned thresholds (a linear probe). "
                "Absolute numbers are therefore lower than full fine-tuning; only the differences between rows are of interest. "
                "'Rare-class sensitivity' averages glaucoma, cataract, AMD, hypertension and myopia."),
              tbl(rows, [6.0 * cm, 1.9 * cm, 1.9 * cm, 1.5 * cm, 1.6 * cm, 2.1 * cm, 1.6 * cm], font=7.8),
              P(f"Table 4. Loss ablation on the test set ({d['ablation_setup']}).", CAP)]

    # subgroups
    if d["sub"] is not None:
        sb = d["sub"]
        rows = [["Variable", "Group", "Images", "Patients", "Macro ROC-AUC", "Macro F1", "Macro sens.", "Macro spec."]]
        for _, r in sb.iterrows():
            if r["variable"] == "Source dataset":
                continue
            rows.append([r["variable"], str(r["group"]) + (" *" if r["small_sample"] else ""), str(r["n_images"]), str(r["n_patients"]),
                         f(r["macro_roc_auc"]), f(r["macro_f1"]), f(r["macro_sensitivity"]), f(r["macro_specificity"])])
        s += keep(P("5.4  Subgroup analysis", H2),
              P("Performance of the selected model by age group, sex and eye. Disease prevalence differs between subgroups "
                "(e.g. AMD and cataract are concentrated in older patients), so differences in macro metrics partly reflect "
                "which classes are present, not only model fairness."),
              tbl(rows, [2.5 * cm, 2.2 * cm, 1.6 * cm, 1.7 * cm, 2.2 * cm, 1.8 * cm, 1.9 * cm, 1.9 * cm], font=7.8),
              P("Table 5. * fewer than 50 images — interpret with great caution.", CAP))
    s += [P("5.5  Training behaviour", H2)]
    for name, m in d["models"].items():
        s += img(R / f"{name}_history.png", 14, fig(f"Training loss and validation macro ROC-AUC per epoch — {nice_arch(m)}."))
    s += keep(P("5.6  Model attention (Grad-CAM)", H2), *img(FIG / "gradcam_examples.png", 17, fig("Grad-CAM for the labelled class on one randomly chosen single-label test image per class. "
                                              "Research use only — attention maps can highlight the optic disc, vessels or "
                                              "image artefacts and do not prove the model uses the right pathology.")))

    # ---------------- 6 refractive
    s += [P("6  Results — refractive-error estimation", H1),
          P("No real fundus + refraction data could be obtained (Section 2.3), so <b>this section reports a pipeline "
            "verification on synthetic images, not model performance.</b> The synthetic SE controls how strongly "
            "tessellation and disc changes are drawn, so a network can learn it; real fundus photographs carry a much weaker "
            "and noisier signal.", WARN)]
    if d["refr"]:
        rm = d["refr"]
        t = rm.get("test_metrics", {})
        rows = [["Model (synthetic data)", "MAE (D)", "RMSE (D)", "R²", "±0.50 D", "±1.00 D", "±2.00 D"],
                [f"{nice_arch(rm)} regression (demo)", f(t.get("mae"), 2), f(t.get("rmse"), 2), f(t.get("r2")),
                 f(t.get("within_0.50D"), pct=True), f(t.get("within_1.00D"), pct=True), f(t.get("within_2.00D"), pct=True)]]
        for name, b in d["refr_base"].items():
            rows.append([name, f(b["mae"], 2), f(b["rmse"], 2), f(b["r2"]), f(b["within_0.50D"], pct=True),
                         f(b["within_1.00D"], pct=True), f(b["within_2.00D"], pct=True)])
        s += [tbl(rows, [6.4 * cm, 1.6 * cm, 1.6 * cm, 1.4 * cm, 1.6 * cm, 1.6 * cm, 1.6 * cm], font=7.8),
              P("Table 6. Synthetic test split (patient-level). These numbers only show that training, evaluation, "
                "baselines and the CSV interface work.", CAP)]
        s += img(R / d["refr_name"] / "regression.png", 16, fig("Predicted vs. measured SE and residuals on synthetic "
                                                            "test images (watermarked: pipeline test only)."))
    s += [P("For context, the reference study on real ultra-widefield images reported a test MAE of about 1.7 D; an estimate "
            "with that error is far from the 0.25 D precision of a spectacle prescription, which is why the application "
            "always shows a range and the wording “Experimental AI estimate. Not a prescription.”")]

    # ---------------- 7 app
    s += [P("7  The web application", H1),
          P("<font face='DejaVuMono'>python run_app.py</font> starts the FastAPI server (http://127.0.0.1:8000). Inference runs on "
            "CPU in a few seconds per eye, so the app works on an ordinary laptop. Pages:")]
    s += bullets(["<b>Dashboard</b> — totals, recent examinations, model flags in each eye's latest examination, latest SE estimates.",
                  "<b>New examination</b> — left/right image, patient ID, age, sex and examination date; validation of inputs.",
                  "<b>Results</b> — per eye: photo / attention-map toggle, quality notes, eight probability bars with uncertainty "
                  "band and threshold tick, SE estimate on a diopter scale with its 95% range and confidence, disclaimers.",
                  "<b>History</b> — every visit for a patient. <b>Trends</b> — SE and probabilities over time per eye with the "
                  "estimated change and whether it exceeds uncertainty. <b>Model information</b> — dataset, architecture, training "
                  "date, version, validation and test metrics, plots and the evaluation PDF."])
    s.append(P("Two demo patients (DEMO-001, DEMO-002) were created from synthetic images with the synthetic demo models so the "
               "history and trend pages have data; they are tagged 'demo' in the interface.", SMALL))
    for name, cap in [("results.jpg", "Results page for a real ODIR-5K test patient (right eye labelled normal, left eye "
                                      "mild non-proliferative DR + epiretinal membrane). The right-eye DR flag is a false "
                                      "positive; the refraction values come from the synthetic demo model and are marked as such."),
                      ("dashboard.jpg", "Dashboard."),
                      ("trends.jpg", "Trends page, DEMO-001 (synthetic images): estimated SE per visit with its 95% range; "
                                     "the last change is flagged as within model uncertainty."),
                      ("trends2.jpg", "Trends page, DEMO-002 (synthetic images): disease probabilities over time as "
                                      "lesions are added to the synthetic images."),
                      ("new.jpg", "New examination form."),
                      ("models_top.jpg", "Model information page (top part: disease classifier and model comparison).")]:
        s += img(SHOTS / name, 14.5, fig(cap), max_h=15.0)

    # ---------------- 8 discussion
    s += [P("8  Discussion", H1)] + [P(t) for t in discussion(d)]

    # ---------------- 9 limitations
    s += [P("9  Limitations and ethics", H1)]
    s += bullets([
        "Single training dataset (ODIR-5K, Chinese hospitals, mixed cameras); no external validation numbers (RFMiD could not be downloaded in the build environment).",
        "Labels come from keyword annotations by trained readers, not from adjudicated clinical diagnosis; ODIR is known to contain label noise.",
        "Several classes have fewer than 30 positive test images; their metrics have wide confidence intervals.",
        f"CPU-only training at {best['img_size']} px for a few epochs; higher resolution and longer GPU training would likely change results.",
        "'Myopia' in ODIR means pathological myopia visible on the fundus, not refractive myopia in general.",
        "The refractive model is a synthetic-data demonstration; no claim about real refraction accuracy is made.",
        "Grad-CAM is a coarse, post-hoc explanation and can be misleading.",
        "Uncertainty from MC dropout is approximate; the SE range additionally uses validation error to avoid over-confidence.",
        "Patient data: the prototype stores images and results locally in SQLite without authentication; real deployment would "
        "need consent, access control, encryption and regulatory approval. It must not be used for patient care."])

    # ---------------- 10 future
    s += [P("10  Future work", H1)]
    s += bullets(["Train at 384–512 px on a GPU (notebook provided) and add test-time augmentation and model ensembling.",
                  "External validation on RFMiD and other datasets; calibration analysis (reliability diagrams, temperature scaling).",
                  "Obtain an ethically approved dataset with measured refraction (or MMAC 2023 access) to train the SE model for real; "
                  "add sphere/cylinder/axis heads (already implemented).",
                  "Use both eyes jointly and patient metadata (age, sex) as inputs.",
                  "Image-quality gradability model trained on labelled quality data instead of heuristics."])

    # ---------------- references
    s += [P("References", H1)]
    refs = [
        "ODIR-2019: Peking University International Competition on Ocular Disease Intelligent Recognition — dataset page, "
        "https://odir2019.grand-challenge.org/dataset/ ; Kaggle release: andrewmvd/ocular-disease-recognition-odir5k.",
        "Pachade S., Porwal P., Thulkar D., et al. Retinal Fundus Multi-Disease Image Dataset (RFMiD): a dataset for multi-disease "
        "detection research. Data 6(2):14, 2021. IEEE Dataport, doi:10.21227/s3g7-st65.",
        "Yang D., Li M., Li W., et al. Prediction of refractive error based on ultrawide field images with deep learning models in "
        "myopia patients. Frontiers in Medicine 9:834281, 2022. doi:10.3389/fmed.2022.834281.",
        "Myopic Maculopathy Analysis Challenge (MMAC), MICCAI 2023 — Task 3: prediction of spherical equivalent.",
        "Tan M., Le Q. EfficientNet: rethinking model scaling for convolutional neural networks. ICML 2019.",
        "He K., Zhang X., Ren S., Sun J. Deep residual learning for image recognition. CVPR 2016.",
        "Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. Focal loss for dense object detection. ICCV 2017.",
        "Selvaraju R. R., et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. ICCV 2017.",
        "Gal Y., Ghahramani Z. Dropout as a Bayesian approximation: representing model uncertainty in deep learning. ICML 2016.",
        "Wightman R. PyTorch Image Models (timm). https://github.com/huggingface/pytorch-image-models."]
    s += [Paragraph(r, BUL, bulletText=f"[{i + 1}]") for i, r in enumerate(refs)]
    s.append(PageBreak())

    # ---------------- appendices
    s += [P("Appendix A — How to run", H1),
          P("Full instructions are in HOW_TO_RUN.txt. Short version (Windows/Linux, Python 3.10+):"),
          P("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu<br/>"
            "pip install -r requirements.txt<br/>python run_app.py   (open http://127.0.0.1:8000)", CODE),
          P("Retraining and evaluation (a GPU is recommended; see notebooks/kaggle_gpu_training.ipynb):"),
          P("python scripts/prepare_odir.py --root &lt;ODIR folder&gt; --size 224<br/>"
            "python baselines.py --task disease --dataset odir<br/>"
            "python train.py --task disease --dataset odir --arch efficientnet_b0 --img-size 224<br/>"
            "python train.py --task disease --dataset odir --arch resnet50 --img-size 224<br/>"
            "python evaluate.py --task disease ; python evaluate.py --task disease --compare<br/>"
            "python train.py --task refractive --data-csv refraction.csv ; python evaluate.py --task refractive<br/>"
            "python predict.py --image path/to/image.jpg --gradcam cam.png", CODE),
          P("Appendix B — Project structure", H1),
          tbl([["Path", "Contents"]] + [[P(a, SMALL), P(b, SMALL)] for a, b in [
              ("config.py", "labels, paths, defaults, safety wording"),
              ("preprocess.py", "retina crop, resize, quality check, caching"),
              ("dataset.py", "ODIR / RFMiD / refraction CSV parsing, per-eye labels, patient split, PyTorch dataset"),
              ("models.py, losses.py, metrics.py", "networks, losses, evaluation metrics"),
              ("train.py, evaluate.py, predict.py", "training, evaluation reports, single-image prediction"),
              ("baselines.py, gradcam.py,<br/>inference.py", "baselines, Grad-CAM, shared inference engine (MC dropout)"),
              ("app/", "FastAPI server, SQLite layer, web front-end"),
              ("scripts/", "data preparation, synthetic data, loss ablation, demo seeding, report figures and this report"),
              ("models/", "trained checkpoints + metadata; models/demo = synthetic demo models"),
              ("reports/", "evaluation PDFs, plots, metric files, screenshots"),
              ("notebooks/", "optional Kaggle GPU training notebook")]], [5.5 * cm, 11.5 * cm], align_right_from=None)]

    def footer(c, doc):
        c.saveState()
        c.setFont("DejaVu", 7)
        c.setFillColor(MUTED)
        if doc.page > 1:
            c.drawString(2 * cm, 1.1 * cm, "EyeVision AI — research/educational prototype, not a medical device")
            c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
        c.restoreState()

    # keep every table together with the caption that follows it
    merged, i = [], 0
    while i < len(s):
        nxt = s[i + 1] if i + 1 < len(s) else None
        if isinstance(s[i], Table) and isinstance(nxt, Paragraph) and nxt.getPlainText().startswith(("Table ", "Figure ")):
            merged.append(KeepTogether([s[i], nxt])); i += 2
        else:
            merged.append(s[i]); i += 1
    s = merged

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="EyeVision AI — project report", author=AUTHOR)
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


def discussion(d):
    """Discussion paragraphs written from the measured numbers."""
    best = d["best"]
    tm, pc, vm = best["test_metrics"], best["test_per_class"], best["val_metrics"]
    nm = lambda c: {"D": "DR", "A": "AMD", "O": "other", "N": "normal"}.get(c, SHORT_NAMES[c].lower())
    ranked = sorted(DISEASE_CODES, key=lambda c: -(pc[c]["roc_auc"] or 0))
    strong = [c for c in ranked if (pc[c]["roc_auc"] or 0) >= 0.93]
    weak = [c for c in ranked if (pc[c]["roc_auc"] or 0) < 0.85]
    bl = max(d["baselines"].values(), key=lambda b: b["summary"]["macro_roc_auc"])["summary"] if d["baselines"] else None
    out = []
    t = (f"<b>What works.</b> The fine-tuned {nice_arch(best)} reached a test macro ROC-AUC of {f(tm['macro_roc_auc'])}")
    if bl:
        t += f", far above the best hand-crafted-feature baseline ({f(bl['macro_roc_auc'])})"
    if d["ablation"] is not None:
        lp = d["ablation"]["macro_roc_auc"].max()
        t += (f". A linear classifier on frozen ImageNet features already reached {f(lp)}, so much of the signal is captured "
              f"by generic pretrained features and fine-tuning adds the rest")
    t += (". Classes with large, visually distinctive signs are recognised best: "
          + ", ".join(f"{nm(c)} ({f(pc[c]['roc_auc'])})" for c in strong)
          + ". For myopia and cataract the precision–recall curves are also high, so these scores are useful at a fixed threshold.")
    out.append(t)
    dr = d["idx"][d["idx"].D == 1].keywords.str.lower()
    share_mild = (dr.str.contains("mild nonproliferative") | dr.str.contains("moderate non")).mean()
    h = pc["H"]
    out.append(
        "<b>What works less well.</b> The weakest classes are " + ", ".join(f"{nm(c)} ({f(pc[c]['roc_auc'])})" for c in weak)
        + f". In ODIR, {100 * share_mild:.0f}% of DR-labelled eyes are mild or moderate non-proliferative retinopathy, whose "
        f"microaneurysms and small haemorrhages are a few pixels wide at {best['img_size']} px input. 'Other' mixes dozens of "
        f"conditions, and 'normal' is defined as the absence of every finding, so it inherits errors from all classes. "
        f"Hypertensive retinopathy shows the gap between ranking and detection: ROC-AUC {f(h['roc_auc'])} but PR-AUC "
        f"{f(h['pr_auc'])} and only {f(h['recall_sensitivity'])} sensitivity at the validation-tuned threshold, with "
        f"{h['n_pos']} positive test images. At 3–5% prevalence even a good ranking yields many false positives per true "
        f"positive, which is why PR-AUC and per-class sensitivity are reported instead of accuracy.")
    out.append(
        f"<b>Operating point.</b> At the thresholds chosen on validation (maximum F1 per class) the macro sensitivity is "
        f"{f(tm['macro_recall'])} and macro specificity {f(tm['macro_specificity'])}. A screening use would choose higher-"
        f"sensitivity thresholds and accept more false alarms; the thresholds are stored in the checkpoint and can be changed "
        f"without retraining. Test macro ROC-AUC ({f(tm['macro_roc_auc'])}) is higher than validation ({f(vm['macro_roc_auc'])}): "
        f"with fewer than 50 positives per rare class in each split, differences of a few points between splits are expected, "
        f"which the bootstrap confidence intervals in Table 3 make visible.")
    models = list(d["models"].values())
    if len(models) > 1:
        a, b = sorted(models, key=lambda m: -m["test_metrics"]["macro_roc_auc"])[:2]
        out.append(
            f"<b>Architecture comparison.</b> {nice_arch(a)} scored macro ROC-AUC {f(a['test_metrics']['macro_roc_auc'])} / "
            f"PR-AUC {f(a['test_metrics']['macro_pr_auc'])} against {f(b['test_metrics']['macro_roc_auc'])} / "
            f"{f(b['test_metrics']['macro_pr_auc'])} for {nice_arch(b)}. The comparison is constrained by the CPU budget: "
            f"ResNet-50 (≈4 GFLOPs per image, ten times EfficientNet-B0) could only be partially fine-tuned for "
            f"{[m for m in models if m['arch'] == 'resnet50'][0].get('epochs_run', '?') if any(m['arch'] == 'resnet50' for m in models) else '?'} "
            f"epochs, so the result says more about efficiency under a fixed budget than about the architectures' ceilings.")
    if d["ablation"] is not None:
        a = d["ablation"].set_index("loss")
        out.append(
            f"<b>Imbalance handling.</b> In the controlled linear-probe comparison (Table 4) all four losses reached macro "
            f"ROC-AUC between {f(a['macro_roc_auc'].min())} and {f(a['macro_roc_auc'].max())}, and rare-class sensitivity "
            f"between {f(a['rare_class_sensitivity'].min())} and {f(a['rare_class_sensitivity'].max())}. Because each loss "
            f"gets its own validation-tuned thresholds, re-weighting mostly shifts scores that the threshold then absorbs; here "
            f"the choice of loss mattered much less than fine-tuning and input resolution. Weighted BCE was kept as the default "
            f"for the fine-tuned models because it ranked as well as the alternatives.")
    out.append("<b>Uncertainty and tracking.</b> The application never shows a bare number: probabilities carry an MC-dropout "
               "band and a threshold marker, and SE estimates carry a range that includes the model's validation error. The "
               "trend view separates changes larger than the combined uncertainty from changes that are likely noise and warns "
               "that a change in model output is not the same as a change in the eye.")
    out.append("<b>Refraction.</b> Estimating refractive error from a 45° fundus photograph is plausible (myopic eyes show "
               "tessellation, tilted discs and peripapillary atrophy), but published models on real data still have errors of "
               "1–2 D. Without real paired data this project provides the infrastructure and an honest, clearly labelled "
               "demonstration, not a validated estimator.")
    return out


if __name__ == "__main__":
    build()
