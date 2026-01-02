# coding: utf-8
import os, pickle, logging
from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm

from .video_preprocessor import get_face_crops

from src.utils.feature_store import (
    FeatureStore, build_cache_key, need_full_reextract, merge_missing
)


class MultimodalDataset(Dataset):
    """
    Multimodal dataset for body, face (later — audio, text, scene).
    Reads CSV, extracts features, caches them (per-modality pickle files).
    Cache is per modality (face/audio/text/behavior); meta stored separately.
    """

    def __init__(
        self,
        csv_path: str,
        video_dir: str,
        audio_dir: str,
        config,
        split: str,
        modality_feature_extractors: dict,
        dataset_name: str,
        device: str = "cuda",
    ):
        super().__init__()

        # ───────── base fields ─────────
        self.csv_path                = csv_path
        self.video_dir               = video_dir
        self.audio_dir               = audio_dir
        self.config                  = config
        self.split                   = split
        self.dataset_name            = dataset_name
        self.device                  = device
        self.segment_length          = config.counter_need_frames
        self.subset_size             = config.subset_size
        self.average_features        = config.average_features
        self.text_description_column = config.text_description_column

        self.face_detector           = config.face_detector
        self.face_rel_thr            = config.face_relative_threshold
        self.average_multi_face      = config.average_multi_face

        # ───────── modality dicts ─────────
        self.extractors: dict[str, object] = modality_feature_extractors

        # ───────── cache setup ─────────
        self.save_prepared_data = config.save_prepared_data
        self.save_feature_path  = config.save_feature_path

        # centralized store for caches/meta
        self.store = FeatureStore(self.save_feature_path)

        # ───── label setup ─────
        if self.dataset_name == 'cmu_mosei':
            self.emotion_columns = [
                "Neutral", "Anger", "Disgust", "Fear",
                "Happiness", "Sadness", "Surprise"
                ]
            self.personality_columns  = []
            self.ah_columns = []

        elif self.dataset_name == 'fiv2':
            self.personality_columns = [
                "openness", "conscientiousness", "extraversion", "agreeableness", "non-neuroticism"
                ]
            self.emotion_columns = []
            self.ah_columns = []
        elif self.dataset_name == 'bah':
            self.ah_columns = ["absence_full", "presence_full"]
            self.emotion_columns = []
            self.personality_columns = []
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        self.num_emotion = 7
        self.num_personality = 5
        self.num_ah = 2

        # ───────── read CSV ─────────
        self.df = pd.read_csv(self.csv_path).dropna()

        if self.subset_size > 0:
            self.df = self.df.head(self.subset_size)
            logging.info(f"[DatasetMultiModal] Using only the first {len(self.df)} records (subset_size={self.subset_size}).")

        self.video_names = sorted(self.df["video_name"].unique())
        self.meta: list[dict] = []

        # meta is stored separately from features
        if self.save_prepared_data:
            self.meta = self.store.load_meta(
                self.dataset_name, self.split, self.config.random_seed, self.subset_size
            )
        else:
            self.meta = []

        if not self.meta:
            # build only the dataset skeleton (paths + labels), without features
            self._build_meta_only()
            if self.save_prepared_data:
                self.store.save_meta(
                    self.dataset_name, self.split, self.config.random_seed, self.subset_size, self.meta
                )

        # incrementally (per-modality) prepare features/cache files
        self._prepare_modality_caches()

    # ────────────────────────── utils ──────────────────────────── #
    def _find_file(self, base_dir: str, base_filename: str):
        for root, _, files in os.walk(base_dir):
            for file in files:
                if os.path.splitext(file)[0] == base_filename:
                    return os.path.join(root, file)
        return None

    def _make_label_dict(
        self,
        emotion:      torch.Tensor | None,
        personality:  torch.Tensor | None,
        ah:           torch.Tensor | None,
    ) -> Dict[str, torch.Tensor | None]:
        """
        Return a dict with all keys present. If a label is missing — use None.
        """
        return {
            "emotion":     emotion,
            "personality": personality,
            "ah": ah
        }

    # ──────────────────────────────────────────────────────────────────
    def _build_meta_only(self):
        self.meta = []
        for name in tqdm(self.video_names, desc="Indexing samples"):
            video_path = self._find_file(self.video_dir, name)
            audio_path = self._find_file(self.audio_dir, name)

            if video_path is None:
                print(f"❌ Video not found: {name}")
                continue
            if audio_path is None:
                print(f"❌ Audio not found: {name}")
                continue

            # ---------- labels ------------------------------------- #
            try:
                emotion_tensor     = None
                personality_tensor = None
                ah_tensor          = None

                #   ─ emotion ─
                if self.emotion_columns:
                    emotion_tensor = torch.tensor(
                        self.df.loc[
                            self.df["video_name"] == name, self.emotion_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    emotion_tensor = torch.full(
                        (self.num_emotion,), torch.nan, dtype=torch.float32
                    )

                #   ─ personality ─
                if self.personality_columns:
                    personality_tensor = torch.tensor(
                        self.df.loc[
                            self.df["video_name"] == name, self.personality_columns
                        ].values[0],
                        dtype=torch.float32
                    )
                else:
                    personality_tensor = torch.full(
                        (self.num_personality,), torch.nan, dtype=torch.float32
                    )

                # AH (BAH)
                if self.ah_columns:
                    ah_vals = self.df.loc[self.df["video_name"] == name, self.ah_columns].values[0]
                    # one-hot → single-label (0/1)
                    ah_tensor = torch.tensor(int(ah_vals[1] == 1), dtype=torch.long)
                else:
                    ah_tensor = torch.tensor(float('nan'))

                labels = self._make_label_dict(
                    emotion_tensor,
                    personality_tensor,
                    ah_tensor
                )

            except Exception as e:
                logging.warning(f"Label extract error {name}: {e}")
                labels = self._make_label_dict(
                    torch.full((self.num_emotion,), torch.nan, dtype=torch.float32),
                    torch.full((self.num_personality,), torch.nan, dtype=torch.float32),
                    torch.tensor(float('nan')),
                )

            # store only the skeleton (no features)
            self.meta.append({
                "sample_name": name,
                "video_path": video_path,
                "audio_path": audio_path,
                "labels": labels,
            })

    # ──────────────────────────────────────────────────────────────────
    def _prepare_modality_caches(self):

        if not self.meta:
            return
        sample_names = [m["sample_name"] for m in self.meta]

        plan: List[tuple[str, Any]] = []
        if "face" in self.extractors:
            plan.append(("face", self.extractors["face"]))
        if "audio" in self.extractors:
            plan.append(("audio", self.extractors["audio"]))
        if "text" in self.extractors:
            plan.append(("text", self.extractors["text"]))
        if "behavior" in self.extractors:
            plan.append(("behavior", self.extractors["behavior"]))

        for mod, ex in plan:
            key = build_cache_key(mod, ex, self.config)
            store, header = self.store.load_modality_store(
                self.dataset_name, self.split, key, self.config.random_seed, self.subset_size
            )

            if need_full_reextract(self.config, mod, header, key):
                store = {}

            missing = merge_missing(store, sample_names)
            if not missing:
                continue

            for name in tqdm(missing, desc=f"Extracting {mod}"):
                try:
                    feats = None

                    if mod == "face":
                        # detection + raw face crops → extractor normalizes internally
                        _, face_images = get_face_crops(
                            video_path=self._find_file(self.video_dir, name),
                            segment_length=self.segment_length,
                            detector=self.face_detector,
                            relative_threshold=self.face_rel_thr,
                            average_multi_face = self.average_multi_face,
                        )
                        feats = ex.extract(images=face_images) if face_images else None

                    elif mod == "audio":
                        ap = self._find_file(self.audio_dir, name)
                        feats = ex.extract(audio_path=ap) if ap else None

                    elif mod == "text":
                        if "text" in self.df.columns:
                            txt_raw = self.df[self.df["video_name"] == name]["text"].values[0]
                            feats = ex.extract(txt_raw)

                    elif mod == "behavior":
                        behavior_col = self.text_description_column
                        if behavior_col in self.df.columns:
                            behavior_txt = self.df[self.df["video_name"] == name][behavior_col].values[0]
                            feats = ex.extract(text=behavior_txt)

                    feats = self._aggregate(feats, self.average_features) if feats is not None else None
                    store[name] = feats  # dict of core keys or None
                except Exception as e:
                    logging.warning(f"{mod} extract error {name}: {e}")
                    store[name] = None

            self.store.save_modality_store(
                self.dataset_name, self.split, key, self.config.random_seed, self.subset_size, store
            )

            # small CUDA memory relief
            torch.cuda.empty_cache()

    def _aggregate(self, feats: Any, average: str) -> Optional[dict]:
        """
        feats: dict with key 'embedding' -> Tensor [T,D] or [D]
        Returns:
          - {'mean','std'} if average='mean_std'
          - {'mean'}       if average='mean'
          - {'seq'}        if average='raw'
        """
        if not isinstance(feats, dict):
            raise TypeError(f"Expected dict with key 'embedding', got {type(feats)}")

        emb = feats.get("embedding", None)
        if emb is None or not isinstance(emb, torch.Tensor):
            raise TypeError(f"Features dict must contain 'embedding' Tensor, got keys {list(feats.keys())}")

        if emb.ndim == 1:   # [D]
            emb = emb.unsqueeze(0)  # [1,D]

        if average == "mean_std":
            return {
                "mean": emb.mean(dim=0),
                "std":  emb.std(dim=0, unbiased=False),
            }
        elif average == "mean":
            return {
                "mean": emb.mean(dim=0)
            }
        else:
            return {"seq": emb}

    # ───────────────────── dataset API ─────────────────────────── #
    def __len__(self):  return len(self.meta)

    def __getitem__(self, idx):
        base = self.meta[idx]
        name = base["sample_name"]

        features = {}
        for mod, ex in self.extractors.items():
            key = build_cache_key(mod, ex, self.config)
            cache = self.store.get_store(
                self.dataset_name, self.split, key, self.config.random_seed, self.subset_size
            )
            features[mod] = cache.get(name, None)

        return {
            **base,
            "features": features,
        }
