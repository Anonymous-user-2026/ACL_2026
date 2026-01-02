from __future__ import annotations
import logging
from typing import Dict, List, Any
import torch
from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from functools import partial

from .dataset_multimodal import MultimodalDataset


# ---------------------------
# Collate & feature stacking
# ---------------------------

def _stack_core_feats(feat_dict: dict, emb_normalize: bool) -> torch.Tensor:
    if "mean" in feat_dict and "std" in feat_dict:
        features = torch.cat([feat_dict["mean"], feat_dict["std"]], dim=0)
    elif "mean" in feat_dict and "std" not in feat_dict:
        features = feat_dict["mean"]
    elif "seq" in feat_dict:
        emb = feat_dict["seq"]
        if emb.ndim == 2:
            features = emb.mean(dim=0)
        elif emb.ndim == 1:
            features = emb
        else:
            raise ValueError(f"Unexpected seq shape: {tuple(emb.shape)}")
    else:
        raise ValueError(f"Unsupported feature dict keys: {list(feat_dict.keys())}")

    if emb_normalize:
        features = features / (features.abs().max() + 1e-8)
    return features


def custom_collate_fn(batch, *, emb_normalize: bool):
    # 1) filter invalid samples
    filtered_batch = [s for s in batch if s and isinstance(s.get("features"), dict)]
    if not filtered_batch:
        return None

    # 2) intersection of actually present modalities
    present_sets = []
    for s in filtered_batch:
        present = [m for m, f in s["features"].items() if isinstance(f, dict)]
        if present:
            present_sets.append(set(present))
    if not present_sets:
        return None
    common_modalities = set.intersection(*present_sets)
    if not common_modalities:
        return None

    # 3) assemble tensors
    features: Dict[str, torch.Tensor] = {}
    for m in sorted(common_modalities):
        core_vecs = []
        for sample in filtered_batch:
            fd = sample["features"].get(m)
            if not isinstance(fd, dict):
                core_vecs = []  # consistency broken — drop this modality
                break
            core_vecs.append(_stack_core_feats(fd, emb_normalize=emb_normalize))
        if core_vecs:
            features[m] = torch.stack(core_vecs, dim=0)

    if not features:
        return None

    emo = torch.stack([b["labels"]["emotion"] for b in filtered_batch])
    per = torch.stack([b["labels"]["personality"] for b in filtered_batch])
    ah  = torch.stack([b["labels"]["ah"] for b in filtered_batch])

    return {
        "features": features,           # dict(modality -> [B, D_total])
        "labels": {
            "emotion": emo,
            "personality": per,
            "ah": ah,
        },
    }


# ---------------------------
# Fractions
# ---------------------------

def _resolve_fraction_for_split(ds_cfg: dict, split: str) -> float:
    """
    Use an explicit fraction per dataset/split:
      - {split}_fraction (0..1), or
      - common fallback: "fraction" (0..1).
    0 → for train/dev/test return an empty dataset.
    """
    key = f"{split}_fraction"
    if key in ds_cfg:
        raw = ds_cfg[key]
    elif "fraction" in ds_cfg:
        raw = ds_cfg["fraction"]
    else:
        raise ValueError(
            f"No fraction for split '{split}'. "
            f"Provide '{key}' (0..1) or a common 'fraction'."
        )
    try:
        f = float(raw)
    except Exception:
        raise ValueError(f"'{key if key in ds_cfg else 'fraction'}' must be a number in 0..1, not {raw!r}.")
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"Fraction must be in the range 0..1, not {f}.")
    return f


class EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0
    def __getitem__(self, idx):
        raise IndexError


# ---------------------------
# Builder
# ---------------------------

def make_dataset_and_loader(
    config,
    split: str,
    modality_extractors: Dict[str, Any],
    *,
    only_dataset: str | None = None,
):
    if not getattr(config, "datasets", None):
        raise ValueError("⛔ [datasets] section is missing in config.")

    datasets: List[torch.utils.data.Dataset] = []

    for dataset_name, ds_cfg in config.datasets.items():
        if only_dataset and dataset_name != only_dataset:
            continue

        frac = _resolve_fraction_for_split(ds_cfg, split)

        if frac == 0.0:
            logging.warning(f"[{dataset_name}:{split}] fraction=0.00 → empty dataset.")
            datasets.append(EmptyDataset())
            continue

        csv_path  = ds_cfg["csv_path"].format(base_dir=ds_cfg["base_dir"], split=split)
        video_dir = ds_cfg["video_dir"].format(base_dir=ds_cfg["base_dir"], split=split)
        audio_dir = ds_cfg["audio_dir"].format(base_dir=ds_cfg["base_dir"], split=split)

        dataset = MultimodalDataset(
            csv_path=csv_path,
            video_dir=video_dir,
            audio_dir=audio_dir,
            config=config,
            split=split,
            modality_feature_extractors=modality_extractors,
            dataset_name=dataset_name,
            device=config.device,
        )

        if 0.0 < frac < 1.0:
            torch.manual_seed(getattr(config, "random_seed", 42))
            n = len(dataset)
            k = max(1, int(n * frac))
            idx = torch.randperm(n)[:k].tolist()
            dataset = Subset(dataset, idx)
            logging.info(f"[{dataset_name}:{split}] fraction={frac:.2f} → {k}/{n} samples")
        else:
            n = len(dataset)
            logging.info(f"[{dataset_name}:{split}] fraction=1.00 → {n}/{n} samples")

        datasets.append(dataset)

    if not datasets:
        if only_dataset:
            logging.warning(f"[{split}] '{only_dataset}': no active datasets → empty dataset.")
            empty_ds = EmptyDataset()
            collate = partial(custom_collate_fn, emb_normalize=config.emb_normalize)
            empty_loader = DataLoader(empty_ds, batch_size=config.batch_size, collate_fn=collate)
            return empty_ds, empty_loader
        raise ValueError(f"⚠️ No datasets found for split='{split}'.")

    full_dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    collate = partial(custom_collate_fn, emb_normalize=config.emb_normalize)

    total_len = len(full_dataset)
    # if dataset is empty → no RandomSampler
    shuffle_flag = (split == "train") and (total_len > 0)

    logging.info(
        f"[{split}] emb_normalize={'ON' if config.emb_normalize else 'OFF'} (applied in collate_fn)"
    )
    if total_len == 0:
        logging.warning(f"[{split}] empty dataset (len=0) → DataLoader with shuffle=False")

    loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle_flag,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    return full_dataset, loader
