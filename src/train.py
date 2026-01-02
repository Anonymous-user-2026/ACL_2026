# coding: utf-8
from __future__ import annotations

import os, logging
import random
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from lion_pytorch import Lion

from .utils.schedulers import SmartScheduler
from .utils.logger_setup import color_metric, color_split
from .utils.measures import mf1, uar, acc_func, ccc, mf1_ah, uar_ah
from .utils.losses import MultiTaskLossWithNaN_v2, MultiTaskLossWithNaN_v3
from .models.models import MultiModalFusionModel_v1, MultiModalFusionModel_v2, MultiModalFusionModel_v3

# ─────────────────────────────── utils ────────────────────────────────
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def transform_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    "Tricky" post-process for emotions under a multiclass metric:
    first logit — "neutral/background", remaining six — real classes.
    """
    threshold1 = 1 - 1/7
    threshold2 = 1/7
    mask1 = matrix[:, 0] >= threshold1
    result = np.zeros_like(matrix[:, 1:])
    transformed = (matrix[:, 1:] >= threshold2).astype(int)
    result[~mask1] = transformed[~mask1]
    return result

def process_predictions(pred_emo: torch.Tensor, true_emo: torch.Tensor):
    """
    Transform emotion predictions for mF1/mUAR:
    softmax -> transform_matrix -> target binarization (drop class 0).
    """
    pred_emo = torch.nn.functional.softmax(pred_emo, dim=1).cpu().detach().numpy()
    pred_emo = transform_matrix(pred_emo).tolist()
    true_emo = true_emo.cpu().detach().numpy()
    true_emo = np.where(true_emo > 0, 1, 0)[:, 1:].tolist()
    return pred_emo, true_emo

# ───────────────────────────── ablation I/O ─────────────────────────────

def drop_domains_in_batch(batch: dict, cfg):
    """Filter out disabled modalities at batch level (strict mode)."""
    ab = cfg.ablation  # either None, or object with fields
    if ab and ab.disabled_modalities and "features" in batch:
        disabled = set(ab.disabled_modalities)
        feats = batch["features"]
        kept = {k: v for k, v in feats.items() if k not in disabled}
        if not kept:
            raise ValueError("All modalities in batch were disabled by ablation; empty batch['features'].")
        batch["features"] = kept
    return batch


def _first_nonempty_batch(loader: DataLoader) -> dict:
    """Return the first non-None batch (since collate_fn may return None)."""
    it = iter(loader)
    while True:
        b = next(it)  # if dataset is empty, fail loudly
        if b is not None:
            return b


def _infer_modal_dims_from_batch(batch: dict[str, Any]) -> dict[str, int]:
    """Look at batch['features'][mod] -> [B, D] and collect {mod: D}."""
    dims = {}
    for mod, x in batch["features"].items():
        if isinstance(x, torch.Tensor):
            dims[mod] = int(x.shape[-1])
            # dims[mod] = int(x.shape[1]/2)
    if not dims:
        raise ValueError("Could not infer modality dimensions from batch.")
    return dims

# ─────────────────────────── evaluation ────────────────────────────
@torch.no_grad()
def evaluate_epoch(model: torch.nn.Module,
                   loader: DataLoader,
                   device: torch.device,
                   cfg) -> Dict[str, float]:
    """Compute metrics on the whole loader (emotion/personality/AH)."""
    model.eval()
    emo_preds, emo_tgts = [], []
    pkl_preds, pkl_tgts = [], []
    ah_preds, ah_tgts = [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        batch = drop_domains_in_batch(batch, cfg)
        out = model(batch)

        # Emotion
        logits_e = out.get("emotion_logits")
        if logits_e is not None:
            y_e = batch["labels"]["emotion"]
            valid_e = ~torch.isnan(y_e).all(dim=1)
            if valid_e.any():
                p, t = process_predictions(logits_e[valid_e], y_e[valid_e])
                emo_preds.extend(p)
                emo_tgts.extend(t)

        # Personality
        preds_p = out.get("personality_scores")
        if preds_p is not None:
            y_p = batch["labels"]["personality"]
            valid_p = ~torch.isnan(y_p).all(dim=1)
            if valid_p.any():
                pkl_preds.append(preds_p[valid_p].detach().cpu().numpy())
                pkl_tgts.append(y_p[valid_p].detach().cpu().numpy())

        # AH (binary, logits->[B,2])
        logits_ah = out.get("ah_logits")
        if logits_ah is not None and "ah" in batch["labels"]:
            y_ah = batch["labels"]["ah"]
            # allow float with NaN or long without NaN
            valid_ah = ~(torch.isnan(y_ah) if y_ah.dtype.is_floating_point else torch.zeros_like(y_ah, dtype=torch.bool))
            if valid_ah.any():
                pred = logits_ah[valid_ah].argmax(dim=1).cpu().numpy()
                tgt = (y_ah[valid_ah].long() if y_ah.dtype != torch.long else y_ah[valid_ah]).cpu().numpy()
                ah_preds.append(pred)
                ah_tgts.append(tgt)

    metrics: Dict[str, float] = {}
    if emo_tgts:
        tgt, prd = np.asarray(emo_tgts), np.asarray(emo_preds)
        metrics["mF1"] = float(mf1(tgt, prd))
        metrics["mUAR"] = float(uar(tgt, prd))
    if pkl_tgts:
        tgt, prd = np.vstack(pkl_tgts), np.vstack(pkl_preds)
        metrics["ACC"] = float(acc_func(tgt, prd))
        metrics["CCC"] = float(ccc(tgt, prd))
    if ah_tgts:
        tgt = np.concatenate(ah_tgts, axis=0)
        prd = np.concatenate(ah_preds, axis=0)
        metrics["MF1_AH"] = float(mf1_ah(tgt, prd))
        metrics["UAR_AH"] = float(uar_ah(tgt, prd))
    return metrics


def log_and_aggregate_split(name: str,
                            loaders: dict[str, DataLoader],
                            model: torch.nn.Module,
                            device: torch.device,
                            cfg) -> dict[str, float]:
    """
    Universal logging + aggregation function for dev/test.
    Now includes AH and provides separate mean_ah + overall mean_all.
    """
    logging.info(f"—— {name} metrics ——")
    all_metrics: dict[str, float] = {}

    for ds_name, loader in loaders.items():
        m = evaluate_epoch(model, loader, device, cfg)
        all_metrics.update({f"{k}_{ds_name}": v for k, v in m.items()})
        msg = " · ".join(color_metric(k, v) for k, v in m.items())
        logging.info(f"[{color_split(name)}:{ds_name}] {msg}")

    mf1s = [v for k, v in all_metrics.items() if k.startswith("mF1_")]
    uars = [v for k, v in all_metrics.items() if k.startswith("mUAR_")]
    accs = [v for k, v in all_metrics.items() if k.startswith("ACC_")]
    cccs = [v for k, v in all_metrics.items() if k.startswith("CCC_")]
    f1_ahs = [v for k, v in all_metrics.items() if k.startswith("MF1_AH_")]
    uar_ahs = [v for k, v in all_metrics.items() if k.startswith("UAR_AH_")]

    if mf1s and uars:
        all_metrics["mean_emo"] = float(np.mean(mf1s + uars))
    if accs and cccs:
        all_metrics["mean_pkl"] = float(np.mean(accs + cccs))
    if f1_ahs and uar_ahs:
        all_metrics["mean_ah"] = float(np.mean(f1_ahs + uar_ahs))

    # overall aggregate across all available sub-metrics
    buckets = []
    for k in ("mean_emo", "mean_pkl", "mean_ah"):
        if k in all_metrics:
            buckets.append(all_metrics[k])
    if buckets:
        all_metrics["mean_all"] = float(np.mean(buckets))

    # compact summary
    if any(k in all_metrics for k in ("mean_emo", "mean_pkl", "mean_ah", "mean_all")):
        summary_parts = []
        for k in ("mean_emo", "mean_pkl", "mean_ah", "mean_all"):
            if k in all_metrics:
                summary_parts.append(color_metric(k, all_metrics[k]))
        logging.info(f"{name} Summary | " + " ".join(summary_parts))

    return all_metrics


# ────────────────────────── main train() ──────────────────────────
MODEL_REGISTRY = {
    "MultiModalFusionModel_v1": MultiModalFusionModel_v1,
    "MultiModalFusionModel_v2": MultiModalFusionModel_v2,
    "MultiModalFusionModel_v3": MultiModalFusionModel_v3,
}


def train(cfg,
          mm_loader: DataLoader,
          dev_loaders: dict[str, DataLoader] | None = None,
          test_loaders: dict[str, DataLoader] | None = None):

    seed_everything(cfg.random_seed)
    device = cfg.device

    sample_batch = _first_nonempty_batch(mm_loader)
    modality_input_dim = _infer_modal_dims_from_batch(sample_batch)

    model_cls = MODEL_REGISTRY[cfg.model_name]
    ab = cfg.ablation

    model = model_cls(
        modality_input_dim=modality_input_dim,
        hidden_dim=cfg.hidden_dim,
        num_heads=cfg.num_transformer_heads,
        out_dim=cfg.out_features,
        dropout=cfg.dropout,
        emo_out_dim=7,
        pkl_out_dim=5,
        ah_out_dim=2,
        device=device,
        ablation_cfg=ab,
    ).to(device)

    if ab is not None:
        logging.info(
            "Ablation: "
            f"use_graph={ab.use_graph} "
            f"use_attention={ab.use_attention} "
            f"use_guidebank={ab.use_guidebank} "
            f"use_task_projectors={ab.use_task_projectors} "
            f"disabled_modalities={ab.disabled_modalities} "
            f"active_tasks={ab.active_tasks or ('emotion','personality','ah')}"
        )

    if cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"⛔ Unknown optimizer: {cfg.optimizer}")
    logging.info(f"⚙️ Optimizer: {cfg.optimizer}, learning rate: {cfg.lr}")

    steps_per_epoch = sum(1 for b in mm_loader if b is not None)
    scheduler = SmartScheduler(
        scheduler_type=cfg.scheduler_type,
        optimizer=optimizer,
        config=cfg,
        steps_per_epoch=steps_per_epoch
    )

    criterion = MultiTaskLossWithNaN_v3(
            personality_loss_type=cfg.pers_loss_type,
            emotion_loss_type=cfg.emotion_loss_type,
            emo_weights=(torch.FloatTensor(
                [5.890161, 7.534918, 11.228363, 27.722221, 1.3049748, 5.6189237, 26.639517]
            ).to(device) if cfg.flag_emo_weight else None),
            ah_weights=None,
            alpha_sup=cfg.alpha_sup,
            w_lr_sup=cfg.w_lr_sup,
            alpha_ssl=cfg.alpha_ssl,
            w_lr_ssl=cfg.w_lr_ssl,
            lambda_ssl=cfg.lambda_ssl,
            w_floor=cfg.w_floor,
            ssl_confidence_threshold_emo_ah=cfg.ssl_confidence_threshold_emo_ah,
            ssl_confidence_threshold_pt=cfg.ssl_confidence_threshold_pt,
            weight_emotion=cfg.weight_emotion,
            weight_personality=cfg.weight_pers,
            weight_ah=cfg.weight_ah,
        ).to(device)


    best_dev, best_test = {}, {}
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(cfg.num_epochs):
        logging.info(f"═══ EPOCH {epoch + 1}/{cfg.num_epochs} ═══")
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_preds_emo, total_targets_emo = [], []
        total_preds_per, total_targets_per = [], []
        total_preds_ah,  total_targets_ah  = [], []

        for batch in tqdm(mm_loader):
            if batch is None:
                continue
            batch = drop_domains_in_batch(batch, cfg)

            emo_labels = batch["labels"].get("emotion");     emo_labels = emo_labels.to(device) if emo_labels is not None else None
            per_labels = batch["labels"].get("personality"); per_labels = per_labels.to(device) if per_labels is not None else None
            ah_labels  = batch["labels"].get("ah", None);    ah_labels  = ah_labels.to(device)  if ah_labels  is not None else None

            valid_emo = (~torch.isnan(emo_labels).all(dim=1)) if emo_labels is not None else None
            valid_per = (~torch.isnan(per_labels).all(dim=1)) if per_labels is not None else None
            if ah_labels is None:
                valid_ah = None
            else:
                valid_ah = ~(torch.isnan(ah_labels) if ah_labels.dtype.is_floating_point
                             else torch.zeros_like(ah_labels, dtype=torch.bool))

            outputs = model(batch)

            loss_labels = {}
            if emo_labels is not None:
                loss_labels["emotion"] = emo_labels
                loss_labels["valid_emo"] = valid_emo
            if per_labels is not None:
                loss_labels["personality"] = per_labels
                loss_labels["valid_per"] = valid_per
            if ah_labels is not None:
                loss_labels["ah"] = ah_labels
                loss_labels["valid_ah"] = valid_ah

            total_task_loss, gn_info = criterion(outputs, loss_labels, model=model, return_details=True)

            total_task_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(batch_level=True)

            if emo_labels is not None:
                bs = emo_labels.shape[0]
            elif per_labels is not None:
                bs = per_labels.shape[0]
            elif ah_labels is not None:
                bs = ah_labels.shape[0]
            else:
                bs = next(iter(batch["features"].values())).shape[0]
            total_loss += float(total_task_loss.item()) * bs
            total_samples += bs

            if outputs.get('emotion_logits') is not None and valid_emo is not None and valid_emo.any():
                preds_emo, targets_emo = process_predictions(outputs['emotion_logits'][valid_emo],
                                                             emo_labels[valid_emo])
                total_preds_emo.extend(preds_emo)
                total_targets_emo.extend(targets_emo)

            if outputs.get('personality_scores') is not None and valid_per is not None and valid_per.any():
                preds_per = outputs['personality_scores'][valid_per]
                targets_per = per_labels[valid_per]
                total_preds_per.extend(preds_per.detach().cpu().numpy().tolist())
                total_targets_per.extend(targets_per.detach().cpu().numpy().tolist())

            if outputs.get('ah_logits') is not None and valid_ah is not None and valid_ah.any():
                pred_ah = outputs['ah_logits'][valid_ah].argmax(dim=1).cpu().numpy().tolist()
                tgt_ah  = (ah_labels[valid_ah].long() if ah_labels.dtype != torch.long else ah_labels[valid_ah]).cpu().numpy().tolist()
                total_preds_ah.extend(pred_ah)
                total_targets_ah.extend(tgt_ah)

        train_loss = total_loss / max(1, total_samples)

        if total_targets_emo:
            mF1_train = mf1(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mUAR_train = uar(np.asarray(total_targets_emo), np.asarray(total_preds_emo))
            mean_emo_train = np.mean([mF1_train, mUAR_train])
        else:
            mF1_train = mUAR_train = mean_emo_train = float('nan')

        if total_targets_per:
            t_per = np.asarray(total_targets_per)
            p_per = np.asarray(total_preds_per)
            acc_train = acc_func(t_per, p_per)
            ccc_vals = []
            for i in range(t_per.shape[1]):
                mask = ~np.isnan(t_per[:, i])
                if mask.sum() == 0: continue
                ccc_vals.append(ccc(t_per[mask, i], p_per[mask, i]))
            ccc_train = float(np.mean(ccc_vals)) if ccc_vals else float('nan')
            mean_pkl_train = np.nanmean([acc_train, ccc_train])
        else:
            acc_train = ccc_train = mean_pkl_train = float('nan')

        if total_targets_ah:
            mf1_ah_train = mf1_ah(np.asarray(total_targets_ah), np.asarray(total_preds_ah))
            uar_ah_train = uar_ah(np.asarray(total_targets_ah), np.asarray(total_preds_ah))
            mean_ah_train = np.mean([mf1_ah_train, uar_ah_train])
        else:
            mf1_ah_train = uar_ah_train = mean_ah_train = float('nan')

        sup_w = gn_info.get("weights_sup", {})
        ssl_w = gn_info.get("weights_ssl", {})
        parts = [
            f"Loss={train_loss:.4f}",
            f"EMO: UAR={mUAR_train:.4f} MF1={mF1_train:.4f} MEAN={mean_emo_train:.4f}",
            f"PKL: ACC={acc_train:.4f} CCC={ccc_train:.4f} MEAN={mean_pkl_train:.4f}",
            f"AH:  UAR={uar_ah_train:.4f} MF1={mf1_ah_train:.4f} MEAN={mean_ah_train:.4f}",
            ("w_sup[emo/per/ah]=({:.3f}/{:.3f}/{:.3f}) | "
             "w_ssl[emo/per/ah]=({:.3f}/{:.3f}/{:.3f})".format(
                 sup_w.get("emo_sup", float('nan')),
                 sup_w.get("per_sup", float('nan')),
                 sup_w.get("ah_sup",  float('nan')),
                 ssl_w.get("emo_ssl", float('nan')),
                 ssl_w.get("per_ssl", float('nan')),
                 ssl_w.get("ah_ssl",  float('nan')),
             ))
        ]
        logging.info(f"[{color_split('TRAIN')}] " + " | ".join(parts))

        cur_dev  = log_and_aggregate_split("Dev",  dev_loaders,  model, device, cfg) if dev_loaders  else {}
        cur_test = log_and_aggregate_split("Test", test_loaders, model, device, cfg) if test_loaders else {}

        cur_eval = cur_dev if cfg.early_stop_on == "dev" else cur_test
        metric_val = None
        for key in ("mean_all", "mean_emo", "mean_pkl", "mean_ah"):
            if key in cur_eval:
                metric_val = cur_eval[key]; break
        if metric_val is None:
            metric_val = -float("inf")

        scheduler.step(metric_val)

        improved = metric_val > best_score
        if improved:
            best_score = metric_val
            best_dev, best_test = cur_dev, cur_test
            patience_counter = 0

            os.makedirs(cfg.checkpoint_dir, exist_ok=True)
            def fmt(x): return f"{x:.4f}" if x is not None else "NA"
            ckpt_name = (
                f"best_ep{epoch + 1}"
                f"_all_{fmt(cur_eval.get('mean_all'))}"
                f"_emo_{fmt(cur_eval.get('mean_emo'))}"
                f"_pkl_{fmt(cur_eval.get('mean_pkl'))}"
                f"_ah_{fmt(cur_eval.get('mean_ah'))}.pt"
            )
            ckpt_path = Path(cfg.checkpoint_dir) / ckpt_name
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"✔ Best model saved: {ckpt_path.name}")
        else:
            patience_counter += 1
            logging.warning(f"No improvement — patience {patience_counter}/{cfg.max_patience}")
            if patience_counter >= cfg.max_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    return best_dev, best_test
