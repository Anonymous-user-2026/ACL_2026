# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import pickle
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Any, Tuple

import torch


# ---------------------------
# 1) Cache key (fingerprint + preprocess)
# ---------------------------

@dataclass(frozen=True)
class CacheKey:
    """
    Descriptor for a specific modality/extractor/preprocess.
    Changing any important parameter → key changes → file is recomputed.
    """
    mod: str                     # "face" | "audio" | "text" | "behavior"
    extractor_fp: str            # extractor.fingerprint(), e.g. "clapa:laion/clap-htsat-fused:pooled"
    avg: str                     # average_features: "mean" | "mean_std" | "raw"
    frames: int                  # counter_need_frames (for video/face)
    img: int                     # image_size (for video/face)
    text_col: Optional[str]      # name of text column (for behavior)
    pre_v: str = "v1"            # version of preprocessing logic (get_metadata, etc.)

    def short_id(self) -> str:
        """
        Human-readable identifier for paths/folders.
        Example: face__clipv-openai-clip-vit-base-patch32__frames30_img224_avg-mean_std_pv-v1
        """
        def _sanitize(s: str) -> str:
            bad = r'\/:*?"<>|'
            for ch in bad:
                s = s.replace(ch, '-')
            s = s.replace(' ', '-')       # spaces → '-'
            while '--' in s:
                s = s.replace('--', '-')  # collapse duplicates
            return s

        parts = [self.mod, self.extractor_fp]
        if self.mod == "face":
            parts.append(f"frames{self.frames}")
            parts.append(f"img{self.img}")
        if self.mod == "behavior" and self.text_col:
            parts.append(f"col-{self.text_col}")
        parts.append(f"avg-{self.avg}")
        parts.append(f"pv-{self.pre_v}")

        human = '__'.join(_sanitize(str(p)) for p in parts if p is not None)
        # Windows has path length limits — trim a bit
        return human[:144]


def build_cache_key(mod: str, extractor: Any, cfg: Any) -> CacheKey:
    """
    Build CacheKey from config and extractor.
    extractor must have .fingerprint(); if not — fall back to class name.
    Preprocess parameters are included ONLY if relevant to the modality.
    """
    # extractor fingerprint or safe fallback
    fp_fn = getattr(extractor, "fingerprint", None)
    extractor_fp = fp_fn() if callable(fp_fn) else type(extractor).__name__

    # common fields
    avg_raw = getattr(cfg, "average_features", "mean_std")
    avg = str(avg_raw).strip().lower()  # "mean" | "mean_std" | "raw"
    pre_v = str(getattr(cfg, "preprocess_version", "v1"))

    # modality-specific fields
    if mod == "face":
        frames = int(getattr(cfg, "counter_need_frames", 30))
        img    = int(getattr(cfg, "image_size", 224))
        text_col = None
        # detector = str(getattr(cfg, "face_detector", "mp_fd")).lower()
        # thr_val  = float(getattr(cfg, "face_relative_threshold", 0.3))
        # amf  = int(getattr(cfg, "average_multi_face", True))
        # thr_str  = f"{thr_val:.3f}".rstrip("0").rstrip(".")
        # thr_tag  = thr_str.replace(".", "")
        # extractor_fp = f"{extractor_fp}__det-{detector}__thr{thr_tag}__amf{amf}"

    elif mod == "behavior":
        frames = 0
        img    = 0
        tc = getattr(cfg, "text_description_column", None)
        text_col = str(tc) if (tc is not None and not isinstance(tc, str)) else tc
    else:  # audio, text, other
        frames = 0
        img    = 0
        text_col = None

    return CacheKey(
        mod=mod,
        extractor_fp=extractor_fp,
        avg=avg,
        frames=frames,
        img=img,
        text_col=text_col,
        pre_v=pre_v,
    )


# ---------------------------
# 2) Features: save/load on disk
# ---------------------------

def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def _atomic_save_pt(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _atomic_save_pickle(obj: Any, path: str):
    tmp = f"{path}.tmp_{os.getpid()}_{int(time.time()*1000)}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


class FeatureStore:
    """
    On-disk cache by modalities.
    Stores:
      - meta: list of samples (without features)
      - feats: {sample_name -> dict(core_key -> Tensor)|None} with header=CacheKey
    API:
      - load_meta/save_meta
      - load_modality_store/save_modality_store
      - get_store (lazy in-memory cache)
    """
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._stores_mem: Dict[Tuple[str, str, str, int, int, str, str], Dict[str, Optional[dict]]] = {}
        # mem_key: (dataset, split, mod, seed, subset, avg, short_id)

    # ------ paths
    def _base_dir(self, dataset: str, split: str) -> str:
        return os.path.join(self.root, dataset, split)

    def meta_path(self, dataset: str, split: str, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        _safe_makedirs(base)
        return os.path.join(base, f"meta_seed{seed}_subset{subset}.pickle")

    def feats_path(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> str:
        base = self._base_dir(dataset, split)
        mod_dir = os.path.join(base, key.mod, key.short_id())  # readable subfolder
        _safe_makedirs(mod_dir)
        fname = f"feats_seed{seed}_subset{subset}_avg-{key.avg}.pt"
        return os.path.join(mod_dir, fname)

    # ------ meta
    def load_meta(self, dataset: str, split: str, seed: int, subset: int) -> list[dict]:
        p = self.meta_path(dataset, split, seed, subset)
        if not os.path.exists(p):
            return []
        with open(p, "rb") as f:
            return pickle.load(f)

    def save_meta(self, dataset: str, split: str, seed: int, subset: int, meta: list[dict]):
        p = self.meta_path(dataset, split, seed, subset)
        _atomic_save_pickle(meta, p)

    # ------ modality
    def load_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Tuple[Dict[str, Optional[dict]], Optional[CacheKey]]:
        p = self.feats_path(dataset, split, key, seed, subset)
        if not os.path.exists(p):
            return {}, None
        obj = torch.load(p, map_location="cpu")
        data = obj.get("data", {}) if isinstance(obj, dict) else obj  # backward compatibility
        header = obj.get("header", None)
        if isinstance(header, dict):
            header = CacheKey(**header)
        return data, header

    def save_modality_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int, store: Dict[str, Optional[dict]]):
        p = self.feats_path(dataset, split, key, seed, subset)
        payload = {"header": asdict(key), "data": store}
        _atomic_save_pt(payload, p)

    # ------ lazy in-memory access for __getitem__
    def get_store(self, dataset: str, split: str, key: CacheKey, seed: int, subset: int) -> Dict[str, Optional[dict]]:
        mem_key = (dataset, split, key.mod, seed, subset, key.avg, key.short_id())
        if mem_key in self._stores_mem:
            return self._stores_mem[mem_key]
        store, _ = self.load_modality_store(dataset, split, key, seed, subset)
        self._stores_mem[mem_key] = store
        return store


# ---------------------------
# 3) Helpers
# ---------------------------

def need_full_reextract(cfg: Any, mod: str, old_header: Optional[CacheKey], new_key: CacheKey) -> bool:
    """
    If keys differ → recompute the current target file.
    Also supports manual force via config.
    Expected: cfg.overwrite_modality_cache: bool, cfg.force_reextract: list[str]
    """
    if getattr(cfg, "overwrite_modality_cache", False):
        return True
    force_list = set(cfg.force_reextract)
    if mod in force_list:
        return True
    return (old_header is None) or (old_header != new_key)


def merge_missing(store: Dict[str, Optional[dict]], sample_names: list[str]) -> list[str]:
    """Return list of sample names not present in store — need to extract them."""
    return [s for s in sample_names if s not in store]
