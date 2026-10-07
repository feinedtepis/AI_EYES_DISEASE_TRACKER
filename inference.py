"""
Inference engine shared by predict.py (CLI) and the web app.

For one fundus image it returns:
  - heuristic image-quality report
  - disease probabilities (mean of Monte-Carlo-dropout passes) + per-class
    uncertainty (std across passes) + a validation-tuned decision threshold
  - estimated spherical equivalent (SE) with an uncertainty interval and a
    'model confidence' = P(|true SE - estimate| <= 1.00 D) under a Gaussian
    whose std combines MC-dropout spread and the validation residual std
  - Grad-CAM heatmap for the highest-scoring non-normal class
All outputs are research estimates (see config.DISCLAIMER).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config
from config import DISEASE_CODES, DISEASE_NAMES
from dataset import build_transforms
from models import load_model, pick_best_checkpoint
from preprocess import preprocess_fundus, quality_check


def _enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def _binary_entropy(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class EyeVisionEngine:
    def __init__(self, device="auto", allow_demo=True, disease_ckpt=None, refractive_ckpt=None):
        self.device = config.resolve_device(device)
        self.disease, self.disease_meta, self.disease_demo = None, None, False
        self.refr, self.refr_meta, self.refr_demo = None, None, False
        p, demo = (Path(disease_ckpt), False) if disease_ckpt else pick_best_checkpoint("disease", allow_demo)
        if p:
            self.disease, self.disease_meta = load_model(p, self.device)
            self.disease_demo = demo or self.disease_meta.get("is_demo", False)
            self.disease_path = str(p)
        p, demo = (Path(refractive_ckpt), False) if refractive_ckpt else pick_best_checkpoint("refractive", allow_demo)
        if p:
            self.refr, self.refr_meta = load_model(p, self.device)
            self.refr_demo = demo or self.refr_meta.get("is_demo", False)
            self.refr_path = str(p)

    # ------------------------------------------------------------------
    def status(self) -> dict:
        def info(meta, demo, path):
            if meta is None:
                return None
            keep = ("model_version", "arch", "dataset", "train_date", "img_size", "loss", "n_train_images",
                    "n_train_patients", "val_metrics", "test_metrics", "external_metrics", "thresholds",
                    "threshold_strategy", "pretrained_imagenet", "residual_std", "epochs_run")
            d = {k: meta.get(k) for k in keep if k in meta}
            d["is_demo"] = bool(demo)
            d["file"] = Path(path).name
            return d
        return {"disease": info(self.disease_meta, self.disease_demo, getattr(self, "disease_path", "")),
                "refractive": info(self.refr_meta, self.refr_demo, getattr(self, "refr_path", "")),
                "device": str(self.device)}

    # ------------------------------------------------------------------
    def analyze(self, image, gradcam_out: str | None = None, mc_passes: int = config.MC_DROPOUT_PASSES) -> dict:
        rgb_q = quality_check(image)
        result = {"quality": rgb_q, "disclaimer": config.DISCLAIMER, "disease": None, "refractive": None,
                  "gradcam": None}

        if self.disease is not None:
            size = int(self.disease_meta.get("img_size", 384))
            sq, mask, _ = preprocess_fundus(image, size)
            x = build_transforms(size, train=False)(Image.fromarray(sq)).unsqueeze(0).to(self.device)
            probs = self._mc(self.disease, x, mc_passes, lambda o: torch.sigmoid(o))
            mean, std = probs.mean(0), probs.std(0)
            th = self.disease_meta.get("thresholds", [0.5] * len(DISEASE_CODES))
            classes = []
            for i, c in enumerate(DISEASE_CODES):
                classes.append({"code": c, "name": DISEASE_NAMES[c], "short": config.SHORT_NAMES[c],
                                "probability": float(mean[i]), "uncertainty": float(std[i]),
                                "threshold": float(th[i]), "above_threshold": bool(mean[i] >= th[i])})
            decisiveness = float(1 - _binary_entropy(mean).mean())
            flagged = [c for c in classes if c["above_threshold"] and c["code"] != "N"]
            result["disease"] = {
                "classes": classes,
                "confidence": decisiveness,
                "confidence_definition": "1 - mean binary entropy of the class probabilities "
                                         "(how decisive the model is; NOT diagnostic accuracy)",
                "flagged_patterns": [f"Elevated model score for {c['name']}-like features "
                                     f"({100 * c['probability']:.0f}%)" for c in flagged],
                "summary": (f"The model's scores are above its threshold for: "
                            f"{', '.join(c['name'] for c in flagged)}. This is not a diagnosis."
                            if flagged else "No class exceeded the model's decision threshold. "
                                            "This does not rule out eye disease."),
                "model_version": self.disease_meta.get("model_version"),
                "is_demo_model": self.disease_demo,
            }
            # Grad-CAM for the highest non-normal class
            if gradcam_out:
                try:
                    from gradcam import GradCAM, overlay_heatmap
                    target = max((c for c in classes if c["code"] != "N"), key=lambda c: c["probability"])
                    idx = DISEASE_CODES.index(target["code"])
                    cam_engine = GradCAM(self.disease)
                    cam = cam_engine(x, idx)
                    cam_engine.remove()
                    over = overlay_heatmap(sq, cam, mask)
                    Path(gradcam_out).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.hstack([sq, over])).save(gradcam_out)
                    result["gradcam"] = {"path": str(gradcam_out), "class": target["name"],
                                         "note": config.GRADCAM_NOTE}
                except Exception as e:     # never fail the whole prediction on the visualisation
                    result["gradcam"] = {"error": str(e), "note": config.GRADCAM_NOTE}

        if self.refr is not None:
            size = int(self.refr_meta.get("img_size", 384))
            sq, _, _ = preprocess_fundus(image, size)
            x = build_transforms(size, train=False)(Image.fromarray(sq)).unsqueeze(0).to(self.device)
            mu, sd = self.refr_meta.get("target_mean", 0.0), self.refr_meta.get("target_std", 1.0)
            se = self._mc(self.refr, x, mc_passes, lambda o: o[:, :1] * sd + mu)[:, 0]
            se_mean, se_mc_std = float(se.mean()), float(se.std())
            resid = float(self.refr_meta.get("residual_std") or 1.0)
            sigma = math.sqrt(se_mc_std ** 2 + resid ** 2)
            tol = config.SE_CONFIDENCE_TOLERANCE_D
            conf = math.erf(tol / (sigma * math.sqrt(2)))
            result["refractive"] = {
                "estimated_se": round(se_mean * 4) / 4,           # displayed in 0.25 D steps
                "estimated_se_raw": se_mean,
                "uncertainty_sd": sigma, "mc_dropout_sd": se_mc_std, "validation_residual_sd": resid,
                "interval_95": [se_mean - 1.96 * sigma, se_mean + 1.96 * sigma],
                "confidence": conf,
                "confidence_definition": f"Estimated probability that the true SE is within ±{tol:.2f} D of the "
                                         f"estimate, using MC-dropout spread + validation error. Experimental.",
                "note": config.SE_NOTE,
                "model_version": self.refr_meta.get("model_version"),
                "is_demo_model": self.refr_demo,
            }
        if not rgb_q["gradable"]:
            result["quality_warning"] = ("This image failed the basic quality checks - "
                                         "model outputs are likely to be unreliable.")
        return result

    @torch.no_grad()
    def _mc(self, model, x, passes, post):
        """Monte-Carlo dropout. Dropout only lives in the heads, so the backbone runs once
        and only the cheap head is sampled `passes` times."""
        model.eval()
        feats = model.backbone(x)
        if passes and passes > 1:
            _enable_mc_dropout(model)
            outs = [post(model.forward_head(feats)).cpu().numpy()[0] for _ in range(passes)]
            model.eval()
            return np.stack(outs)
        return post(model.forward_head(feats)).cpu().numpy()
