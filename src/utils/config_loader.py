# utils/config_loader.py

import os
import toml
import logging
from types import SimpleNamespace

class ConfigLoader:
    """
    Class for loading and processing configuration from `config.toml`.
    """

    def __init__(self, config_path="config.toml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file `{config_path}` not found!")

        self.config = toml.load(config_path)

        # ---------------------------
        # General parameters
        # ---------------------------
        general_cfg = self.config.get("general", {})
        self.use_telegram = general_cfg.get("use_telegram", False)

        # ---------------------------
        # Dataset paths
        # ---------------------------
        self.datasets = self.config.get("datasets", {})

        # ---------------------------
        # DataLoader
        # ---------------------------
        dataloader_cfg = self.config.get("dataloader", {})
        self.num_workers = dataloader_cfg.get("num_workers", 0)
        self.shuffle = dataloader_cfg.get("shuffle", True)
        self.prepare_only = dataloader_cfg.get("prepare_only", False)
        self.text_description_column = dataloader_cfg.get("text_description_column", "text_llm")

        # ---------------------------
        # Training: general
        # ---------------------------
        train_general = self.config.get("train", {}).get("general", {})
        self.random_seed = train_general.get("random_seed", 42)
        self.subset_size = train_general.get("subset_size", 0)
        self.batch_size = train_general.get("batch_size", 8)
        self.num_epochs = train_general.get("num_epochs", 100)
        self.max_patience = train_general.get("max_patience", 10)
        self.save_best_model = train_general.get("save_best_model", False)
        self.save_prepared_data = train_general.get("save_prepared_data", True)
        self.save_feature_path = train_general.get("save_feature_path", "./features/")
        self.search_type = train_general.get("search_type", "none")
        self.early_stop_on = train_general.get("early_stop_on", "dev")
        self.checkpoint_dir = train_general.get("checkpoint_dir","checkpoints")
        self.device = train_general.get("device", "cuda")
        self.selection_metric = train_general.get("selection_metric", "mean_combo")

        # ---------------------------
        # Training: model parameters
        # ---------------------------
        train_model = self.config.get("train", {}).get("model", {})
        self.model_name = train_model.get("model_name", "MyExperiment")
        self.per_activation = train_model.get("per_activation", "sigmoid")
        self.hidden_dim = train_model.get("hidden_dim", 256)
        self.num_transformer_heads = train_model.get("num_transformer_heads", 8)
        self.positional_encoding = train_model.get("positional_encoding", True)
        self.dropout = train_model.get("dropout", 0.15)
        self.out_features = train_model.get("out_features", 128)

        # ---------------------------
        # Training: losses
        # ---------------------------
        train_losses = self.config.get("train", {}).get("losses", {})
        self.weight_emotion = train_losses.get("weight_emotion", 1.0)
        self.weight_pers = train_losses.get("weight_pers", 1.0)
        self.weight_ah = train_losses.get("weight_ah", 1.0)
        self.ssl_weight_emotion = train_losses.get("ssl_weight_emotion", 1)
        self.ssl_weight_personality = train_losses.get("ssl_weight_personality", 1)
        self.ssl_weight_ah = train_losses.get("ssl_weight_ah", 1)
        self.ssl_confidence_threshold_emo_ah = train_losses.get("ssl_confidence_threshold_emo_ah", 0.6)
        self.ssl_confidence_threshold_pt = train_losses.get("ssl_confidence_threshold_pt", 0.6)
        self.pers_loss_type = train_losses.get("pers_loss_type", "mae")
        self.emotion_loss_type = train_losses.get("emotion_loss_type", "CE")
        self.flag_emo_weight = train_losses.get("flag_emo_weight", False)
        self.alpha_sup = train_losses.get("alpha_sup", 1.0)
        self.w_lr_sup = train_losses.get("w_lr_sup", 0.025)
        self.alpha_ssl = train_losses.get("alpha_ssl", 0.5)
        self.w_lr_ssl = train_losses.get("w_lr_ssl", 0.001)
        self.lambda_ssl = train_losses.get("lambda_ssl", 0.2)
        self.w_floor = train_losses.get("w_floor", 1e-3)


        # ---------------------------
        # Training: optimizer
        # ---------------------------
        train_optimizer = self.config.get("train", {}).get("optimizer", {})
        self.optimizer = train_optimizer.get("optimizer", "adam")
        self.lr = train_optimizer.get("lr", 1e-4)
        self.weight_decay = train_optimizer.get("weight_decay", 0.0)
        self.momentum = train_optimizer.get("momentum", 0.9)

        # ---------------------------
        # Training: scheduler
        # ---------------------------
        train_scheduler = self.config.get("train", {}).get("scheduler", {})
        self.scheduler_type = train_scheduler.get("scheduler_type", "plateau")
        self.warmup_ratio = train_scheduler.get("warmup_ratio", 0.1)

        # ---------------------------
        # Embeddings
        # ---------------------------
        emb_cfg = self.config.get("embeddings", {})
        self.average_features = emb_cfg.get("average_features", "mean_std")
        self.emb_normalize = emb_cfg.get("emb_normalize", True)
        self.video_extractor = emb_cfg.get("video_extractor", "off")
        self.audio_extractor = emb_cfg.get("audio_extractor", "off")
        self.text_extractor = emb_cfg.get("text_extractor", "off")
        self.behavior_extractor = emb_cfg.get("behavior_extractor", "off")
        self.counter_need_frames = emb_cfg.get("counter_need_frames", 20)
        self.image_size = emb_cfg.get("image_size", 224)
        self.image_size = emb_cfg.get("image_size", 224)
        self.face_detector = emb_cfg.get("face_detector", "mp_fd")
        self.face_relative_threshold = emb_cfg.get("face_relative_threshold", 0.3)
        self.average_multi_face = emb_cfg.get("average_multi_face", True)

        # ---------------------------
        # Cache
        # ---------------------------
        cache_cfg = self.config.get("cache", {})
        self.per_modality_cache = cache_cfg.get("per_modality_cache", True)
        self.overwrite_modality_cache = cache_cfg.get("overwrite_modality_cache", False)
        self.force_reextract = cache_cfg.get("force_reextract", [])
        self.preprocess_version = cache_cfg.get("preprocess_version", "v1")

        # ---------------------------
        # Ablation
        # ---------------------------

        ab = self.config.get("ablation", None)
        if ab is None:
            self.ablation = None
        else:
            self.ablation = SimpleNamespace(
                use_graph          = ab.get("use_graph", None),
                use_attention      = ab.get("use_attention", None),
                use_guidebank      = ab.get("use_guidebank", None),
                use_task_projectors = ab.get("use_task_projectors", None),
                disabled_modalities= ab.get("disabled_modalities", []),
                active_tasks       = ab.get("active_tasks", None),
            )

        if __name__ == "__main__":
            self.log_config()

    def log_config(self):
        logging.info("=== CONFIGURATION ===")
        logging.info(f"Datasets loaded: {list(self.datasets.keys())}")
        for name, ds in self.datasets.items():
            logging.info(f"[Dataset: {name}]")
            logging.info(f"  Base Dir: {ds.get('base_dir', 'N/A')}")
            logging.info(f"  CSV Path: {ds.get('csv_path', '')}")
            logging.info(f"  WAV Dir: {ds.get('wav_dir', 'N/A')}")
            logging.info(f"  Video Dir: {ds.get('video_dir', '')}")
            logging.info(f"  Audio Dir: {ds.get('audio_dir', '')}")

        # Log training parameters
        logging.info("--- Training Config ---")
        logging.info(f"DataLoader: batch_size={self.batch_size}, num_workers={self.num_workers}, shuffle={self.shuffle}")
        logging.info(f"Model Name: {self.model_name}")
        logging.info(f"Random Seed: {self.random_seed}")
        logging.info(f"Hidden Dim: {self.hidden_dim}")
        logging.info(f"Num Heads in Transformer: {self.num_transformer_heads}")
        logging.info(f"Optimizer: {self.optimizer}")
        logging.info(f"Scheduler Type: {self.scheduler_type}")
        logging.info(f"Warmup Ratio: {self.warmup_ratio}")
        logging.info(f"Weight Decay for Adam: {self.weight_decay}")
        logging.info(f"Momentum (SGD): {self.momentum}")
        logging.info(f"Positional Encoding: {self.positional_encoding}")
        logging.info(f"Dropout: {self.dropout}")
        logging.info(f"Out Features: {self.out_features}")
        logging.info(f"LR: {self.lr}")
        logging.info(f"Num Epochs: {self.num_epochs}")
        logging.info(f"Max Patience: {self.max_patience}")
        logging.info(f"Save Prepared Data: {self.save_prepared_data}")
        logging.info(f"Path to Save Features: {self.save_feature_path}")
        logging.info(f"Search Type: {self.search_type}")
        logging.info(f"Video Extractor: {self.video_extractor}")
        logging.info(f"Audio Extractor: {self.audio_extractor}")
        logging.info(f"Text Extractor: {self.text_extractor}")
        logging.info(f"Behavior Extractor: {self.behavior_extractor}")

    def show_config(self):
        self.log_config()
