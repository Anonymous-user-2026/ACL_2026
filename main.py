# coding: utf-8
import logging
import os
import shutil
import datetime
import toml
import requests
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from src.utils.config_loader import ConfigLoader
from src.utils.logger_setup import setup_logger
from src.utils.search_utils import greedy_search, exhaustive_search
from src.data_loading.dataset_builder import make_dataset_and_loader
from src.data_loading.pretrained_extractors import build_extractors_from_config

from src.train import train

# ───────────────────── optionally load .env ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ───────────────────── Telegram helper ─────────────────────
def _notify_telegram(text: str, enabled: bool = True) -> bool:
    """Sends a message to TG if enabled and TELEGRAM_BOT_TOKEN/CHAT_ID are set.
       Returns True/False and logs the reason for silence."""
    if not enabled:
        logging.info("TG notify: disabled by config")
        return False
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.info("TG notify: skipped (no TELEGRAM_BOT_TOKEN/CHAT_ID)")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=8,
        )
        # Log what Telegram responded with
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}
        if r.ok and isinstance(payload, dict) and payload.get("ok"):
            logging.info("TG notify: sent")
            return True
        logging.warning(f"TG notify: API error {r.status_code} -> {payload}")
        return False
    except Exception as e:
        logging.warning(f"TG notify failed: {e}")
        return False

def main():
    # ──────────────────── 1. Config and directories ────────────────────
    base_config = ConfigLoader("config.toml")

    model_name = base_config.model_name.replace("/", "_").replace(" ", "_").lower()
    timestamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/results_{model_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    base_config.checkpoint_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(base_config.checkpoint_dir, exist_ok=True)

    epochlog_dir = os.path.join(results_dir, "metrics_by_epoch")
    os.makedirs(epochlog_dir, exist_ok=True)

    # ──────────────────── 2. Logging ────────────────────────────
    log_file = os.path.join(results_dir, "session_log.txt")
    setup_logger(logging.INFO, log_file=log_file)
    base_config.show_config()

    use_tg = base_config.use_telegram
    logging.info(f"use_telegram = {use_tg}  (env token={bool(os.getenv('TELEGRAM_BOT_TOKEN'))}, chat={bool(os.getenv('TELEGRAM_CHAT_ID'))})")

    # startup ping — handy to confirm everything is connected
    _notify_telegram(f"🚀 Start: <b>{model_name}</b>\n📁 {results_dir}", enabled=use_tg)

    shutil.copy("config.toml", os.path.join(results_dir, "config_copy.toml"))
    overrides_file = os.path.join(results_dir, "overrides.txt")
    csv_prefix     = os.path.join(epochlog_dir, "metrics_epochlog")

    # ──────────────────── 3. Extractors ─────────────────
    logging.info("🔧 Initializing extractors from config...")
    modality_extractors = build_extractors_from_config(base_config)
    enabled = ", ".join(sorted(modality_extractors.keys())) or "—"
    logging.info(f"✅ Enabled modalities: {enabled}")

    # ──────────────────── 4. Dataloaders ────────────────────────────
    train_datasets = []
    train_loaders, dev_loaders, test_loaders = {}, {}, {}

    for dataset_name in tqdm(base_config.datasets, desc="Dataloaders", leave=False):
        logging.info(f"📦 Loading dataset: {dataset_name}")

        # --- TRAIN: build ONCE, store both ds and loader ---
        train_ds, train_loader = make_dataset_and_loader(
            base_config, "train",
            modality_extractors,
            only_dataset=dataset_name,
        )

        # If fraction = 0 → train_ds length is 0 (EmptyDataset). Don't include in union.
        if len(train_ds) == 0:
            logging.info(f"[train] {dataset_name}: fraction=0 → skipping from union")
        else:
            train_datasets.append(train_ds)
            train_loaders[dataset_name] = train_loader

        # --- DEV (or fallback to VAL) ---
        dev_split = "dev" if os.path.exists(
            base_config.datasets[dataset_name]["csv_path"].format(
                base_dir=base_config.datasets[dataset_name]["base_dir"],
                split="dev",
            )
        ) else "val"

        dev_ds, dev_loader = make_dataset_and_loader(
            base_config, dev_split,
            modality_extractors,
            only_dataset=dataset_name,
        )
        dev_loaders[dataset_name] = dev_loader

        # --- TEST (fallback to dev if test is missing) ---
        test_split_path = base_config.datasets[dataset_name]["csv_path"].format(
            base_dir=base_config.datasets[dataset_name]["base_dir"],
            split="test",
        )
        if os.path.exists(test_split_path):
            test_ds, test_loader = make_dataset_and_loader(
                base_config, "test",
                modality_extractors,
                only_dataset=dataset_name,
            )
            test_loaders[dataset_name] = test_loader
        else:
            test_loaders[dataset_name] = dev_loader  # fallback

    # ──────────────────── 5. prepare_only ──────────────────────────
    if base_config.prepare_only:
        logging.info("== prepare_only mode: only data preparation, no training ==")
        _notify_telegram(
            f"✅ <b>{model_name}</b>: prepare_only completed\n📁 {results_dir}",
            enabled=use_tg
        )
        return

    if len(train_datasets) == 0:
        raise ValueError("🚫 Empty train: all datasets have train_fraction=0. Increase fractions in config.datasets.*")

    union_train_ds = ConcatDataset(train_datasets)

    # take collate_fn from any already created train_loader
    any_train_loader = next(iter(train_loaders.values()))
    union_train_loader = DataLoader(
        union_train_ds,
        batch_size=base_config.batch_size,
        shuffle=True,
        num_workers=base_config.num_workers,
        collate_fn=any_train_loader.collate_fn,
    )

    logging.info(f"[union:train] total samples = {len(union_train_ds)}")

    # ──────────────────── 6. Hyperparameter search / single run ──
    search_config = toml.load("search_params.toml")
    param_grid = dict(search_config["grid"])
    default_values = dict(search_config["defaults"])

    if base_config.search_type == "greedy":
        greedy_search(
            base_config    = base_config,
            train_loader   = union_train_loader,
            dev_loader     = dev_loaders,
            test_loader    = test_loaders,
            train_fn       = train,
            overrides_file = overrides_file,
            param_grid     = param_grid,
            default_values = default_values,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: greedy search finished\n📁 {results_dir}",
            enabled=use_tg
        )

    elif base_config.search_type == "exhaustive":
        exhaustive_search(
            base_config    = base_config,
            train_loader   = union_train_loader,
            dev_loader     = dev_loaders,
            test_loader    = test_loaders,
            train_fn       = train,
            overrides_file = overrides_file,
            param_grid     = param_grid,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: exhaustive search finished\n📁 {results_dir}",
            enabled=use_tg
        )

    elif base_config.search_type == "none":
        logging.info("== Single training mode (no hyperparameter search) ==")
        train(
            cfg              = base_config,
            mm_loader        = union_train_loader,
            dev_loaders      = dev_loaders,
            test_loaders     = test_loaders,
        )
        _notify_telegram(
            f"✅ <b>{model_name}</b>: training (no search) completed\n📁 {results_dir}",
            enabled=use_tg
        )

    else:
        raise ValueError(
            f"⛔️ Invalid search_type in config: '{base_config.search_type}'. "
            f"Use 'greedy', 'exhaustive' or 'none'."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # crash notification always goes out so you know everything burned down
        _notify_telegram(
            f"❌ Crash: <code>{type(e).__name__}</code>\n{e}",
            enabled=True
        )
        raise
