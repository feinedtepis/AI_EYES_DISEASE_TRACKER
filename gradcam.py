"""
Grad-CAM (Selvaraju et al., 2017) for the disease classifier and the
refractive regressor.

IMPORTANT: a Grad-CAM heatmap shows which image regions most influenced the
model's output for one class. It does NOT prove the model is looking at the
correct pathology. Label: "Model attention visualization — research use only."
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models import gradcam_layer


class GradCAM:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer = gradcam_layer(model)
        self._acts = None
        self._grads = None
        self._h = self.layer.register_forward_hook(self._fwd)

    def _fwd(self, module, inp, out):
        self._acts = out
        if out.requires_grad:          # plain no-grad forward passes are left alone
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def remove(self):
        self._h.remove()

    def __call__(self, x: torch.Tensor, output_index: int) -> np.ndarray:
        """x: (1,3,H,W). output_index: class index (disease) or 0 (SE regression)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = self.model(x)
            out[0, output_index].backward()
        acts, grads = self._acts.detach()[0], self._grads.detach()[0]        # (C,h,w)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * acts).sum(0))
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, (x.shape[-1], x.shape[-2]), interpolation=cv2.INTER_CUBIC)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, mask: np.ndarray | None = None, alpha=0.45) -> np.ndarray:
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    cam = np.clip(cam, 0, 1)
    if mask is not None:
        cam = cam * (cv2.resize(mask, (rgb.shape[1], rgb.shape[0])) > 0)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (rgb.astype(np.float32) * (1 - alpha * cam[..., None]) + heat.astype(np.float32) * alpha * cam[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)
