"""
Training entry point.

    python train.py --task disease    --dataset odir --data-root data/ODIR-5K --arch efficientnet_b0
    python train.py --task disease    --dataset odir --arch resnet50 --loss focal
    python train.py --task refractive --data-csv data/refraction.csv --arch resnet50
    python train.py --task disease    --dataset synthetic --img-size 192 --no-pretrained   (pipeline test)

Patient-level train/val/test split is saved in data/splits/ and re-used so
every architecture / loss is compared on the SAME held-out patients.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config
from config import DISEASE_CODES, TrainConfig
from dataset import FundusDataset, get_split, load_index, split_name_for
from losses import build_loss
from metrics import multilabel_metrics, regression_metrics, tune_thresholds
from models import (build_model, early_bn_eval, freeze_early, model_filename, new_model_version,
                    save_checkpoint, set_backbone_trainable)


def parse_args() -> TrainConfig:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description="Train the disease classifier or the refractive-error regressor")
    ap.add_argument("--task", choices=["disease", "refractive"], required=True)
    ap.add_argument("--dataset", default=None, help="odir | rfmid | odir+rfmid | csv | synthetic")
    ap.add_argument("--data-root", default="", help="dataset folder (first run builds the index)")
    ap.add_argument("--data-csv", default="", help="refractive CSV, or custom disease index CSV")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--img-size", type=int, default=d.img_size)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--loss", default=None, help="disease: bce | wbce | focal ; refractive: smoothl1 | mse")
    ap.add_argument("--focal-gamma", type=float, default=d.focal_gamma)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--freeze-epochs", type=int, default=d.freeze_epochs)
    ap.add_argument("--dropout", type=float, default=d.dropout)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--num-workers", type=int, default=d.num_workers)
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--no-hflip", action="store_true", help="disable horizontal flip augmentation")
    ap.add_argument("--multihead", action="store_true", help="refractive: also train sphere/cyl/axis heads")
    ap.add_argument("--freeze-early", action="store_true",
                    help="CPU-friendly: keep the first ~half of the backbone frozen for the whole run")
    ap.add_argument("--max-samples", type=int, default=0, help="debug: use only N images")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tag", default="", help="suffix for the saved model file name")
    ap.add_argument("--out-dir", default=str(config.MODELS_DIR))
    a = ap.parse_args()
    cfg = TrainConfig(
        task=a.task,
        dataset=a.dataset or ("odir" if a.task == "disease" else "csv"),
        arch=a.arch or ("efficientnet_b0" if a.task == "disease" else "resnet50"),
        img_size=a.img_size, batch_size=a.batch_size, epochs=a.epochs, lr=a.lr,
        loss=a.loss or ("wbce" if a.task == "disease" else "smoothl1"), focal_gamma=a.focal_gamma,
        pretrained=not a.no_pretrained, freeze_epochs=a.freeze_epochs, dropout=a.dropout,
        patience=a.patience, num_workers=a.num_workers, seed=a.seed, hflip=not a.no_hflip,
        multihead=a.multihead, max_samples=a.max_samples, device=a.device,
        data_csv=a.data_csv, data_root=a.data_root, tag=a.tag)
    cfg.out_dir = a.out_dir  # type: ignore[attr-defined]
    cfg.freeze_early = a.freeze_early  # type: ignore[attr-defined]
    return cfg


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    outs, ys, ms = [], [], []
    for x, y, m, _ in loader:
        o = model(x.to(device, non_blocking=True)).float().cpu()
        outs.append(o); ys.append(y); ms.append(m)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy(), torch.cat(ms).numpy()


def evaluate_outputs(task, out, y, m, target_mean, target_std, thresholds=None):
    if task == "disease":
        prob = 1 / (1 + np.exp(-out))
        th = thresholds or [0.5] * len(DISEASE_CODES)
        return multilabel_metrics(y, prob, m, th, DISEASE_CODES, ci=False)["summary"], prob
    keep = m[:, 0] > 0
    se_pred = out[keep, 0] * target_std + target_mean
    se_true = y[keep, 0] * target_std + target_mean
    return regression_metrics(se_true, se_pred, ci=False), se_pred


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    device = config.resolve_device(cfg.device)
    print(f"Device: {device}  |  task={cfg.task} arch={cfg.arch} dataset={cfg.dataset} "
          f"img={cfg.img_size} loss={cfg.loss} pretrained={cfg.pretrained}")

    df = load_index(cfg.task, cfg.dataset, cfg.data_root, cfg.data_csv)
    split_name = split_name_for(cfg.task, cfg.dataset, cfg.data_csv)
    df = get_split(df, cfg.task, split_name, cfg)
    if cfg.max_samples:      # debug: keep a random subset of PATIENTS (split assignment unchanged)
        pats = df.patient_id.drop_duplicates()
        keep = pats.sample(frac=min(1.0, cfg.max_samples / len(df)), random_state=0)
        df = df[df.patient_id.isin(keep)]
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print(f"Images: train {len(tr)} | val {len(va)} | test {(df.split == 'test').sum()}  "
          f"(patients: {tr.patient_id.nunique()}/{va.patient_id.nunique()}/"
          f"{df[df.split == 'test'].patient_id.nunique()})")
    if cfg.task == "disease":
        print("Train prevalence:", {c: round(float(tr[c].mean()), 3) for c in DISEASE_CODES})

    t_mean = float(tr["spherical_equivalent"].mean()) if cfg.task == "refractive" else 0.0
    t_std = float(tr["spherical_equivalent"].std() or 1.0) if cfg.task == "refractive" else 1.0

    ds_tr = FundusDataset(tr, cfg.task, cfg.img_size, True, cfg.hflip, t_mean, t_std)
    ds_va = FundusDataset(va, cfg.task, cfg.img_size, False, cfg.hflip, t_mean, t_std)
    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                       pin_memory=pin, drop_last=len(ds_tr) > cfg.batch_size)
    dl_va = DataLoader(ds_va, cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers, pin_memory=pin)

    model = build_model(cfg.task, cfg.arch, cfg.pretrained, cfg.dropout, cfg.multihead).to(device)
    criterion = build_loss(cfg, tr, DISEASE_CODES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, epochs=cfg.epochs,
                                                steps_per_epoch=max(1, len(dl_tr)), pct_start=0.15)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    name = model_filename(cfg.task, cfg.arch, cfg.tag)
    out_dir = Path(cfg.out_dir)
    ckpt_path = out_dir / f"{name}.pth"
    version = new_model_version(cfg.task, cfg.arch, cfg.dataset)
    better = (lambda new, best: new > best) if cfg.task == "disease" else (lambda new, best: new < best)
    best = -1.0 if cfg.task == "disease" else float("inf")
    history, bad_epochs = [], 0

    for epoch in range(1, cfg.epochs + 1):
        frozen = epoch <= cfg.freeze_epochs and cfg.pretrained
        set_backbone_trainable(model, not frozen)
        if cfg.freeze_early:
            freeze_early(model)
        model.train()
        if cfg.freeze_early:
            early_bn_eval(model)
        t0, run_loss, nb = time.time(), 0.0, 0
        for x, y, m, _ in dl_tr:
            x, y, m = x.to(device, non_blocking=True), y.to(device), m.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
            loss = criterion(out.float(), y, m)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item(); nb += 1
        out, y, m = predict_loader(model, dl_va, device)
        val_loss = float(criterion(torch.tensor(out).to(device), torch.tensor(y).to(device), torch.tensor(m).to(device)))
        summ, _ = evaluate_outputs(cfg.task, out, y, m, t_mean, t_std)
        key = summ.get("macro_roc_auc") if cfg.task == "disease" else summ.get("mae")
        key = float(np.nan_to_num(key, nan=0.0 if cfg.task == "disease" else 1e9))
        rec = {"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_loss": val_loss,
               ("val_macro_auc" if cfg.task == "disease" else "val_mae"): key,
               "frozen_backbone": frozen, "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"Epoch {epoch:02d}/{cfg.epochs}  train_loss {rec['train_loss']:.4f}  val_loss {val_loss:.4f}  "
              f"{'val macro-AUC' if cfg.task == 'disease' else 'val MAE (D)'} {key:.4f}  "
              f"({rec['seconds']}s){'  [backbone frozen]' if frozen else ''}")
        if better(key, best):
            best, bad_epochs = key, 0
            save_checkpoint(ckpt_path, model, {"task": cfg.task, "arch": cfg.arch, "partial": True})
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping (no improvement for {cfg.patience} epochs).")
                break

    # ---- reload best weights, calibrate on validation, write final checkpoint ----
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    out, y, m = predict_loader(model, dl_va, device)
    meta = {
        "task": cfg.task, "arch": cfg.arch, "img_size": cfg.img_size, "dropout": cfg.dropout,
        "multihead": cfg.multihead, "model_version": version, "dataset": cfg.dataset,
        "data_csv": cfg.data_csv, "split_name": split_name, "seed": cfg.seed,
        "train_date": dt.datetime.now().isoformat(timespec="seconds"),
        "pretrained_imagenet": cfg.pretrained, "loss": cfg.loss, "epochs_run": len(history),
        "n_train_images": int(len(tr)), "n_val_images": int(len(va)),
        "n_train_patients": int(tr.patient_id.nunique()), "config": cfg.to_dict(),
        "is_demo": cfg.dataset == "synthetic", "freeze_early": cfg.freeze_early,
    }
    if cfg.task == "disease":
        prob = 1 / (1 + np.exp(-out))
        th = tune_thresholds(y, prob, m, "f1")
        vm = multilabel_metrics(y, prob, m, th, DISEASE_CODES, ci=False)
        meta.update({"labels": DISEASE_CODES, "label_names": config.DISEASE_NAMES, "thresholds": th,
                     "threshold_strategy": "max-F1 on validation set",
                     "val_metrics": vm["summary"], "val_per_class": vm["per_class"]})
        print("Validation macro ROC-AUC: %.3f | macro PR-AUC: %.3f | macro F1 (tuned thr): %.3f"
              % (vm["summary"]["macro_roc_auc"], vm["summary"]["macro_pr_auc"], vm["summary"]["macro_f1"]))
    else:
        keep = m[:, 0] > 0
        se_pred = out[keep, 0] * t_std + t_mean
        se_true = y[keep, 0] * t_std + t_mean
        vm = regression_metrics(se_true, se_pred, ci=False)
        meta.update({"target_mean": t_mean, "target_std": t_std,
                     "residual_std": vm.get("residual_std") or vm.get("rmse"), "val_metrics": vm})
        print("Validation MAE %.3f D | RMSE %.3f D | R² %.3f | within ±0.5D %.1f%% | within ±1.0D %.1f%%"
              % (vm["mae"], vm["rmse"], vm["r2"], 100 * vm["within_0.50D"], 100 * vm["within_1.00D"]))
    save_checkpoint(ckpt_path, model, meta)
    hist_path = config.REPORTS_DIR / f"{name}_history.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    _plot_history(history, cfg, config.REPORTS_DIR / f"{name}_history.png")
    print(f"\nSaved model -> {ckpt_path}\nNext: python evaluate.py --task {cfg.task} --checkpoint {ckpt_path}")


def _plot_history(history, cfg, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = pd.DataFrame(history)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ax[0].plot(h.epoch, h.train_loss, label="train"); ax[0].plot(h.epoch, h.val_loss, label="val")
        ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
        col = "val_macro_auc" if cfg.task == "disease" else "val_mae"
        ax[1].plot(h.epoch, h[col], color="#2a7"); ax[1].set_xlabel("epoch")
        ax[1].set_title("Validation macro ROC-AUC" if cfg.task == "disease" else "Validation MAE (D)")
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    except Exception as e:
        print("(could not plot history:", e, ")")


if __name__ == "__main__":
    main()
