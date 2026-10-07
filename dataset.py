"""
Dataset handling: index building for ODIR-5K, RFMiD, refractive-error CSVs and
the synthetic demo set; leakage-free patient-level splitting; PyTorch datasets.

Every source is converted into ONE common per-eye index format (a CSV):

    image_path, patient_id, eye (L/R/U), age, sex (M/F/U), source,
    N, D, G, C, A, H, M, O          <- disease labels (1/0, empty = unknown)
    sphere, cylinder, axis, spherical_equivalent   <- refractive labels (optional)

Disease labels and refractive labels are never mixed: a disease-only dataset
simply has empty refractive columns and is never used for regression, and a
refractive CSV without disease labels is never used for classification.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import DISEASE_CODES

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
REFRACTIVE_COLS = ["sphere", "cylinder", "axis", "spherical_equivalent"]


# =============================================================================
# ODIR-5K
# =============================================================================
# ODIR's N..O columns are PATIENT-level (union of both eyes).  Training on them
# per eye would label a healthy left eye as "cataract" if only the right eye has
# a cataract.  We therefore derive PER-EYE labels from the per-eye diagnostic
# keywords, and use the patient-level columns only as a consistency check.
_ODIR_EXCLUDE = ("no fundus image", "anterior segment image",
                 "optic disk photographically invisible", "low image quality")
_ODIR_IGNORE = ("lens dust", "image offset")
_ODIR_RULES = [
    ("D", re.compile(r"diabetic retinopathy|(non ?)?proliferative( diabetic)? retinopathy")),
    ("G", re.compile(r"glaucoma")),
    ("C", re.compile(r"cataract")),
    ("A", re.compile(r"age-related macular degeneration|age related macular degeneration")),
    ("H", re.compile(r"hypertensive retinopathy")),
    ("M", re.compile(r"myopi")),            # pathological myopia, myopic retinopathy/maculopathy
]


def odir_keywords_to_labels(text: str) -> tuple[dict, bool, list]:
    """Map one eye's ODIR diagnostic keywords to labels. Returns (labels, exclude, unmatched)."""
    labels = {c: 0 for c in DISEASE_CODES}
    if not isinstance(text, str) or not text.strip():
        return labels, True, []
    parts = [p.strip().lower() for p in re.split(r"[，,;；]", text) if p.strip()]
    exclude, normal, unmatched = False, False, []
    for kw in parts:
        if any(x in kw for x in _ODIR_EXCLUDE):
            exclude = True
            continue
        if any(x in kw for x in _ODIR_IGNORE):
            continue
        if "normal fundus" in kw:
            normal = True
            continue
        hit = False
        for code, rx in _ODIR_RULES:
            if rx.search(kw):
                labels[code] = 1
                hit = True
        if not hit:                           # drusen, epiretinal membrane, laser spot, ...
            labels["O"] = 1
            unmatched.append(kw)
    disease = any(labels[c] for c in DISEASE_CODES if c != "N")
    if normal and not disease:
        labels["N"] = 1
    if not normal and not disease:            # e.g. only "lens dust" -> label unknown
        exclude = True
    return labels, exclude, unmatched


def _find_files(root: Path) -> dict:
    files = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            # prefer originals over Kaggle's 'preprocessed_images' copies
            if p.name not in files or "preprocessed" in str(files[p.name]).lower():
                files[p.name] = p
    return files


def build_odir_index(root: str) -> pd.DataFrame:
    root = Path(root)
    ann = None
    for pat in ("*Annotation*.xlsx", "data.xlsx", "*.xlsx", "full_df.csv"):
        hits = sorted(root.rglob(pat))
        if hits:
            ann = hits[0]
            break
    if ann is None:
        raise FileNotFoundError(f"No ODIR annotation file (.xlsx or full_df.csv) found under {root}")
    raw = pd.read_excel(ann) if ann.suffix == ".xlsx" else pd.read_csv(ann)
    raw.columns = [c.strip() for c in raw.columns]
    if "ID" not in raw.columns:
        raise ValueError(f"Unexpected ODIR annotation columns in {ann}: {list(raw.columns)[:12]}")
    raw = raw.drop_duplicates("ID")           # full_df.csv repeats each patient twice
    files = _find_files(root)
    rows, n_excl, n_missing, agree, total = [], 0, 0, 0, 0
    for _, r in raw.iterrows():
        eye_labels = {}
        for eye, fcol, kcol in (("L", "Left-Fundus", "Left-Diagnostic Keywords"),
                                ("R", "Right-Fundus", "Right-Diagnostic Keywords")):
            fname = str(r.get(fcol, f"{r['ID']}_{'left' if eye == 'L' else 'right'}.jpg"))
            labels, exclude, _ = odir_keywords_to_labels(r.get(kcol, ""))
            eye_labels[eye] = labels
            if fname not in files:
                n_missing += 1
                continue
            if exclude:
                n_excl += 1
                continue
            sex = str(r.get("Patient Sex", "U"))[:1].upper()
            rows.append({"image_path": str(files[fname]), "patient_id": f"odir_{r['ID']}", "eye": eye,
                         "age": r.get("Patient Age", np.nan), "sex": sex if sex in "MF" else "U",
                         "source": "odir", "keywords": r.get(kcol, ""), **labels})
        # consistency check with ODIR's own patient-level label columns
        if all(c in raw.columns for c in DISEASE_CODES):
            total += 1
            union = {c: max(eye_labels["L"][c], eye_labels["R"][c]) for c in DISEASE_CODES if c != "N"}
            agree += all(int(r[c]) == union[c] for c in union)
    df = pd.DataFrame(rows)
    print(f"ODIR: {len(df)} eye images from {df.patient_id.nunique()} patients "
          f"(excluded {n_excl} low-quality/unlabelled, {n_missing} files not found).")
    if total:
        print(f"ODIR: per-eye keyword labels agree with official patient-level labels "
              f"for {agree}/{total} patients ({100 * agree / total:.1f}%).")
    return df


# =============================================================================
# RFMiD
# =============================================================================
# RFMiD has 46 condition columns; only some map onto our label space.
# Unmappable targets are left EMPTY (unknown) and masked out of loss/metrics.
RFMID_MAP = {"DR": "D", "ARMD": "A", "MYA": "M", "ODC": "G"}   # ODC = optic disc cupping (glaucoma *proxy*)
RFMID_UNKNOWN = ("C", "H")                                      # no equivalent RFMiD column


def build_rfmid_index(root: str) -> pd.DataFrame:
    root = Path(root)
    csvs = [p for p in root.rglob("*.csv") if "label" in p.name.lower() or "groundtruth" in p.name.lower()]
    if not csvs:
        raise FileNotFoundError(f"No RFMiD label CSVs found under {root}")
    rows = []
    for csv in csvs:
        part = ("train" if "train" in csv.name.lower() else
                "val" if ("eval" in csv.name.lower() or "valid" in csv.name.lower()) else "test")
        # official layout: <Set>/<labels>.csv next to a folder of <ID>.png images
        files = {p.stem: p for p in csv.parent.rglob("*") if p.suffix.lower() in IMG_EXTS}
        lab = pd.read_csv(csv)
        lab.columns = [c.strip() for c in lab.columns]
        cond_cols = [c for c in lab.columns if c not in ("ID", "Disease_Risk")]
        for _, r in lab.iterrows():
            img = files.get(str(int(r["ID"])))
            if img is None:
                continue
            labels = {c: 0.0 for c in DISEASE_CODES}
            for src, dst in RFMID_MAP.items():
                if src in lab.columns:
                    labels[dst] = float(r[src])
            for c in RFMID_UNKNOWN:
                labels[c] = np.nan
            labels["N"] = float(1 - int(r.get("Disease_Risk", 0)))
            other = [c for c in cond_cols if c not in RFMID_MAP and r[c] == 1]
            labels["O"] = float(len(other) > 0)
            rows.append({"image_path": str(img), "patient_id": f"rfmid_{part}_{int(r['ID'])}",
                         "eye": "U", "age": np.nan, "sex": "U", "source": "rfmid",
                         "rfmid_split": part, **labels})
    df = pd.DataFrame(rows)
    print(f"RFMiD: {len(df)} images indexed. NOTE: RFMiD has no patient IDs, so each image is "
          f"treated as its own patient; C and H are unknown (masked).")
    return df


# =============================================================================
# Refractive-error CSV
# =============================================================================
def load_refractive_csv(csv_path: str, data_root: str = "") -> pd.DataFrame:
    """CSV columns: image_path, patient_id, age, sex, sphere, cylinder, axis, spherical_equivalent
    (eye optional). SE is computed as sphere + cylinder/2 where missing."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "image_path" not in df.columns or "patient_id" not in df.columns:
        raise ValueError("Refractive CSV needs at least image_path and patient_id columns")
    for c in REFRACTIVE_COLS + ["age"]:
        if c not in df.columns:
            df[c] = np.nan
    se_missing = df["spherical_equivalent"].isna() & df["sphere"].notna()
    df.loc[se_missing, "spherical_equivalent"] = (df.loc[se_missing, "sphere"]
                                                  + df.loc[se_missing, "cylinder"].fillna(0) / 2.0)
    df = df[df["spherical_equivalent"].notna()].copy()
    base = Path(data_root) if data_root else csv_path.parent
    df["image_path"] = [p if Path(p).is_absolute() else str((base / p).resolve()) for p in df["image_path"]]
    if "eye" not in df.columns:
        df["eye"] = "U"
    df["eye"] = df["eye"].fillna("U").astype(str).str.upper().str[:1]
    if "sex" not in df.columns:
        df["sex"] = "U"
    df["sex"] = df["sex"].fillna("U").astype(str).str.upper().str[:1]
    df["patient_id"] = df["patient_id"].astype(str)
    df["source"] = df.get("source", "csv")
    exists = df["image_path"].map(lambda p: Path(p).exists())
    if (~exists).any():
        print(f"WARNING: {(~exists).sum()} image files listed in the CSV do not exist and are dropped.")
    return df[exists].reset_index(drop=True)


# =============================================================================
# Loading an index for a task
# =============================================================================
def index_path(name: str) -> Path:
    return config.DATA_DIR / f"{name}_index.csv"


def load_index(task: str, dataset: str, data_root: str = "", data_csv: str = "") -> pd.DataFrame:
    if task == "refractive":
        if dataset == "synthetic":
            data_csv = data_csv or str(config.DATA_DIR / "synthetic" / "refractive_labels.csv")
        if not data_csv:
            raise SystemExit(
                "Refractive training needs a CSV with measured refraction:\n"
                "  python train.py --task refractive --data-csv path/to/refraction.csv\n"
                "Columns: image_path,patient_id,age,sex,sphere,cylinder,axis,spherical_equivalent\n"
                "(No large public fundus+refraction dataset is freely downloadable - see README.)")
        return load_refractive_csv(data_csv, data_root)

    parts = dataset.split("+")
    frames = []
    for name in parts:
        if name == "synthetic":
            frames.append(pd.read_csv(config.DATA_DIR / "synthetic" / "disease_labels.csv"))
        elif name == "csv":
            frames.append(pd.read_csv(data_csv))
        elif name in ("odir", "rfmid"):
            p = index_path(name)
            if not p.exists():
                if not data_root:
                    raise SystemExit(f"{name} index not found. Run: python scripts/prepare_{name}.py --root <dataset folder>")
                df = build_odir_index(data_root) if name == "odir" else build_rfmid_index(data_root)
                df.to_csv(p, index=False)
            frames.append(pd.read_csv(p))
        else:
            raise SystemExit(f"Unknown dataset '{name}'")
    df = pd.concat(frames, ignore_index=True)
    for c in DISEASE_CODES:
        if c not in df.columns:
            df[c] = np.nan
    df["patient_id"] = df["patient_id"].astype(str)
    return df


# =============================================================================
# Patient-level split (prevents left/right-eye leakage)
# =============================================================================
def patient_split(df: pd.DataFrame, task: str, val_frac=0.15, test_frac=0.15, seed=42) -> pd.DataFrame:
    """Assign train/val/test per PATIENT.  Stratified on a per-patient key:
    disease -> the rarest positive label of the patient; refractive -> SE quintile."""
    rng = np.random.default_rng(seed)
    pat = df.groupby("patient_id")
    if task == "disease":
        prev = df[DISEASE_CODES].mean()               # NaN-aware prevalence
        order = prev.sort_values().index.tolist()     # rarest first

        def key(g):
            pos = g[DISEASE_CODES].max()
            for c in order:
                if pos.get(c, 0) == 1:
                    return c
            return "none"
        strata = pat.apply(key)
    else:
        se = pat["spherical_equivalent"].mean()
        strata = pd.qcut(se.rank(method="first"), q=min(5, max(1, len(se) // 10)), labels=False)
    assign = {}
    for _, ids in strata.groupby(strata):
        ids = list(ids.index)
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, pid in enumerate(ids):
            assign[pid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    out = df.copy()
    out["split"] = out["patient_id"].map(assign)
    # hard guarantee: no patient appears in two splits
    assert out.groupby("patient_id")["split"].nunique().max() == 1, "patient leakage across splits!"
    return out


def split_name_for(task: str, dataset: str, data_csv: str = "") -> str:
    """Stable name of the saved split file, shared by train.py, baselines.py and evaluate.py."""
    if task == "disease" or dataset == "synthetic":
        return dataset
    return f"{dataset}_{Path(data_csv).stem or 'refraction'}"


def get_split(df: pd.DataFrame, task: str, name: str, cfg) -> pd.DataFrame:
    """Load a saved split if present (so every model is evaluated on the SAME test set)."""
    split_dir = config.DATA_DIR / "splits"
    split_dir.mkdir(exist_ok=True)
    f = split_dir / f"{task}_{name.replace('+', '_')}_seed{cfg.seed}.csv"
    if f.exists():
        saved = pd.read_csv(f, dtype={"patient_id": str})
        m = dict(zip(saved.patient_id, saved.split))
        df = df.copy()
        df["split"] = df["patient_id"].map(m)
        new = df["split"].isna()
        if new.any():
            print(f"{new.sum()} images belong to patients not in the saved split -> assigned to train")
            df.loc[new, "split"] = "train"
    else:
        df = patient_split(df, task, cfg.val_frac, cfg.test_frac, cfg.seed)
        df[["patient_id", "split"]].drop_duplicates().to_csv(f, index=False)
        print(f"Saved patient-level split -> {f}")
    return df


# =============================================================================
# PyTorch datasets / transforms
# =============================================================================
def build_transforms(img_size: int, train: bool, hflip: bool = True):
    from torchvision import transforms as T
    norm = T.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    if not train:
        return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    aug = [T.RandomResizedCrop(img_size, scale=(0.88, 1.0), ratio=(0.95, 1.05))]
    if hflip:
        aug.append(T.RandomHorizontalFlip(0.5))
    aug += [T.RandomRotation(12),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.0),
            T.ToTensor(), norm]
    return T.Compose(aug)


class FundusDataset:
    """Returns (image_tensor, target, mask, row_index). mask=0 marks unknown labels."""

    def __init__(self, df: pd.DataFrame, task: str, img_size: int, train: bool, hflip=True,
                 target_mean=0.0, target_std=1.0):
        import torch  # noqa: F401  (import here so the module loads without torch)
        self.df = df.reset_index(drop=True)
        self.task = task
        self.img_size = img_size
        self.tf = build_transforms(img_size, train, hflip)
        self.target_mean, self.target_std = target_mean, target_std
        cache_col = next((c for c in (f"cached_{img_size}", "cached_384", "cached_512")
                          if c in self.df.columns), None)
        self.paths = [(r[cache_col] if cache_col and isinstance(r[cache_col], str) and r[cache_col]
                       else r["image_path"]) for _, r in self.df.iterrows()]
        self.is_cached = [cache_col is not None and isinstance(r.get(cache_col), str) and bool(r.get(cache_col))
                          for _, r in self.df.iterrows()]

    def __len__(self):
        return len(self.df)

    def _image(self, i):
        from PIL import Image
        from preprocess import preprocess_fundus
        if self.is_cached[i]:
            return Image.open(self.paths[i]).convert("RGB")
        sq, _, _ = preprocess_fundus(self.paths[i], max(self.img_size, 256))
        return Image.fromarray(sq)

    def __getitem__(self, i):
        import torch
        x = self.tf(self._image(i))
        r = self.df.iloc[i]
        if self.task == "disease":
            y = r[DISEASE_CODES].to_numpy(dtype=np.float32)
            m = (~np.isnan(y)).astype(np.float32)
            y = np.nan_to_num(y)
        else:
            vals = r.reindex(REFRACTIVE_COLS).to_numpy(dtype=np.float32)
            se = (vals[3] - self.target_mean) / self.target_std
            # targets: [se_std, sphere, cylinder, cos2θ, sin2θ]
            ax = np.deg2rad(vals[2]) * 2 if not np.isnan(vals[2]) else np.nan
            y = np.array([se, vals[0], vals[1], np.cos(ax), np.sin(ax)], np.float32)
            m = (~np.isnan(y)).astype(np.float32)
            y = np.nan_to_num(y)
        return x, torch.from_numpy(y), torch.from_numpy(m), i
