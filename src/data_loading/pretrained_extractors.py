# coding: utf-8
from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch, torch.nn as nn
import torchaudio
from transformers import (
    CLIPModel, CLIPProcessor,
    ClapModel, ClapProcessor,
    AutoModel, AutoTokenizer, AutoProcessor
)

def _ensure_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    d = (device or "cpu").lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        try:
            return torch.device(d)
        except Exception:
            return torch.device("cuda")
    return torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────
# VIDEO EXTRACTORS
# ─────────────────────────────────────────────────────────────────────
class ClipVideoExtractor:
    """
    Input: images = List[np.ndarray RGB (H,W,3)] or a single np.ndarray.
    Output: {"embedding": Tensor [T, D]} or None if input is empty.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda", output_mode: str = "seq"):
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = CLIPProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"clipv:{self.model_name}"

    @torch.no_grad()
    def extract(self, *, images=None, **_) -> Dict[str, torch.Tensor] | None:
        # Normalize input to a list of RGB images
        if images is None:
            return None
        if isinstance(images, np.ndarray):
            images = [images]
        if not isinstance(images, (list, tuple)) or len(images) == 0:
            return None

        # Processor handles resize/normalize
        batch = self.proc(images=list(images), return_tensors="pt")
        pv = batch["pixel_values"]

        # Shape check
        if pv.ndim == 4 and pv.shape[1] != 3:
            logging.warning(
                f"[ClipVideoExtractor] pixel_values has shape {tuple(pv.shape)}, "
                f"expected [T,3,H,W]. Preprocessing might be wrong."
            )

        pv = pv.to(self.device)
        emb = self.model.get_image_features(pixel_values=pv)  # [T, D] in CLIP space
        return {"embedding": emb}

# ─────────────────────────────────────────────────────────────────────
# AUDIO EXTRACTORS
# ─────────────────────────────────────────────────────────────────────

def load_wav_mono(audio_path: str, target_sr: int) -> np.ndarray:
    if not audio_path:
        raise ValueError("audio_path is empty")
    wav, sr = torchaudio.load(audio_path)         # [C, N]
    if wav.numel() == 0 or wav.shape[-1] == 0:
        raise ValueError(f"empty audio: {audio_path}")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)       # mono
    if sr != int(target_sr):
        wav = torchaudio.functional.resample(wav, sr, int(target_sr))
    wav = wav.squeeze(0)
    wav = wav / (wav.abs().max() + 1e-8)
    return wav
    # return wav.squeeze(0).cpu().numpy().astype("float32")  # [N]

class ClapAudioExtractor:
    """
    Audio → CLAP → tokens [T, H] or pooled [1, D]. Output normalized to [T, H].
    """
    def __init__(self, model_name: str = "laion/clap-htsat-fused",
                 device: str = "cuda", output_mode: str = "seq"):
        self.model_name  = model_name
        self.device      = _ensure_device(device)
        self.output_mode = output_mode  # "seq" | "pooled"

        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.proc  = ClapProcessor.from_pretrained(model_name)

        cfg = getattr(self.model, "config", None)
        sr_from_cfg = getattr(getattr(cfg, "audio_config", None), "sampling_rate", None)
        self.sample_rate = int(sr_from_cfg or 48000)  # source of truth — the model
        logging.info(
            f"[ClapAudioExtractor] Using sample_rate={self.sample_rate} "
            f"(model='{self.model_name}', from_config={sr_from_cfg is not None})"
        )

        self.hidden_sizes = {
            s for s in (
                getattr(getattr(cfg, "audio_config", None), "hidden_size", None),
                getattr(getattr(cfg, "text_config",  None), "hidden_size", None),
                256, 512, 768, 1024, 1536
            ) if isinstance(s, int)
        }

    def fingerprint(self) -> str:
        return f"clapa:{self.model_name}"

    @torch.no_grad()
    def extract(self, *, audio_path: str, **_) -> Dict[str, torch.Tensor]:
        wav_np = load_wav_mono(audio_path, self.sample_rate)

        inputs = self.proc(audios=[wav_np], return_tensors="pt", sampling_rate=self.sample_rate)
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        if self.output_mode == "pooled":
            feats = self.model.get_audio_features(**inputs)  # [1, D]
            return {"embedding": feats}

        args = {}
        if "input_features" in inputs: args["input_features"] = inputs["input_features"]
        if "is_longer" in inputs:      args["is_longer"]      = inputs["is_longer"]

        out = self.model.audio_model(**args, output_hidden_states=True, return_dict=True)
        hidden = out.last_hidden_state
        if hidden is None:
            raise RuntimeError("CLAP audio_model returned no last_hidden_state")

        seq = self._to_T_H(hidden)  # [T, H]
        return {"embedding": seq}

    def _to_T_H(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 4:                          # [B, H, T, F]
            hidden = hidden.mean(dim=-1).transpose(1, 2)  # → [B, T, H]
        elif hidden.ndim == 3:
            B, M, N = hidden.shape
            if (M in self.hidden_sizes) and (N > M or N not in self.hidden_sizes):
                hidden = hidden.transpose(1, 2)       # [B, T, H]
        else:
            raise RuntimeError(f"Unexpected ndim={hidden.ndim} for CLAP hidden")
        return hidden.squeeze(0).contiguous()         # [T, H]

class HFAudioWav2Vec2:
    """
    HF Wav2Vec2-compatible audio extractor.
    Example: "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    Returns:
      - output_mode="seq"    → {"embedding": Tensor[T, H]}
      - output_mode="pooled" → {"embedding": Tensor[1, H]} (time-averaged)
    """
    def __init__(self, model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                 device: str = "cuda", output_mode: str = "seq"):
        self.device      = _ensure_device(device)
        self.output_mode = output_mode
        self.model_name  = model_name

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device).eval()

        sr_from_proc = getattr(getattr(self.processor, "feature_extractor", None), "sampling_rate", None)
        if sr_from_proc is None:
            sr_from_proc = getattr(self.processor, "sampling_rate", None)
        self.sample_rate = int(sr_from_proc or 16000)

        logging.info(
            f"[HFAudioWav2Vec2] Using sample_rate={self.sample_rate} "
            f"(model='{self.model_name}', from_processor={sr_from_proc is not None})"
        )

    def fingerprint(self) -> str:
        return f"hf-w2v2:{self.model_name}"

    @torch.no_grad()
    def extract(self, *, audio_path: str, **_) -> Dict[str, torch.Tensor]:
        wav_np = load_wav_mono(audio_path, self.sample_rate)

        inputs = self.processor(wav_np, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None and getattr(out, "hidden_states", None):
            hidden = out.hidden_states[-1]
        if hidden is None:
            raise RuntimeError(f"{self.model_name}: model output has no last_hidden_state/hidden_states")

        seq = hidden.squeeze(0).contiguous()  # [T, H]
        if self.output_mode == "pooled":
            return {"embedding": seq.mean(dim=0, keepdim=True)}  # [1, H]
        return {"embedding": seq}

# ─────────────────────────────────────────────────────────────────────
# TEXT EXTRACTORS
# ─────────────────────────────────────────────────────────────────────

class ClipTextExtractor:
    """Text/behavior → CLIP text → sequence hidden states or pooled."""
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda", output_mode: str = "seq"):
        if CLIPModel is None or CLIPProcessor is None:
            raise ImportError("transformers with CLIP is not installed. pip install transformers")
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "seq" | "pooled"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"clipt:{self.model_name}"

    @torch.no_grad()
    def extract(self, text: Optional[str] = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if text is None and len(args) > 0 and isinstance(args[0], str):
            text = args[0]
        if text is None:
            text = ""

        inputs = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.output_mode == "pooled":
            feats = self.model.get_text_features(**inputs)     # [1, D]
            return {"embedding": feats}

        out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=False,
            return_dict=True,
        )
        seq = out.last_hidden_state  # [B, L, D]
        if "attention_mask" in inputs:
            valid_len = int(inputs["attention_mask"].sum(dim=1).item())  # B=1
            seq = seq[:, :valid_len, :]
        seq = seq.squeeze(0).contiguous()  # [L, D]
        return {"embedding": seq}

class ClapTextExtractor:
    """Text → CLAP text → sequence hidden states or pooled."""
    def __init__(self, model_name: str = "laion/clap-htsat-fused", device: str = "cuda", output_mode: str = "seq"):
        if ClapModel is None or ClapProcessor is None:
            raise ImportError("transformers with CLAP is not installed. pip install transformers")
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "seq" | "pooled"
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = ClapProcessor.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"clapt:{self.model_name}"

    @torch.no_grad()
    def extract(self, text: Optional[str] = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if text is None and len(args) > 0 and isinstance(args[0], str):
            text = args[0]
        if text is None:
            text = ""
        inputs = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if self.output_mode == "pooled":
            feats = self.model.get_text_features(**inputs)     # [1, D]
            return {"embedding": feats}

        out = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            output_hidden_states=False,
            return_dict=True,
        )
        seq = out.last_hidden_state  # [B, L, D]
        if "attention_mask" in inputs:
            valid_len = int(inputs["attention_mask"].sum(dim=1).item())
            seq = seq[:, :valid_len, :]
        seq = seq.squeeze(0).contiguous()  # [L, D]
        return {"embedding": seq}

class ETC_TextExtractor:
    """Text → michellejieli/emotion_text_classifier → sequence hidden states or pooled embedding."""
    def __init__(self, model_name: str = "michellejieli/emotion_text_classifier", device: str = "cuda", output_mode: str = "pooled"):
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("transformers with AutoModel is not installed. pip install transformers")
        self.model_name = model_name
        self.device = _ensure_device(device)
        self.output_mode = output_mode  # "seq" | "pooled"
        self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def fingerprint(self) -> str:
        return f"emotion-text:{self.model_name}"

    @torch.no_grad()
    def extract(self, text: Optional[str] = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        # Extract text from args if not provided via keyword
        if text is None:
            if args and isinstance(args[0], str):
                text = args[0]
            else:
                text = ""

        # Tokenize input text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get model outputs
        outputs = self.model(**inputs, return_dict=True)

        if self.output_mode == "pooled":
            # Use CLS token embedding
            embeddings = outputs.last_hidden_state[:, 0]
            return {"embedding": embeddings.squeeze(0)}

        # Return full sequence output
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs.get("attention_mask")

        if attention_mask is not None:
            # Remove padding tokens based on attention mask
            valid_len = int(attention_mask.sum(dim=1).item())  # B=1
            last_hidden_state = last_hidden_state[:, :valid_len, :]

        return {"embedding": last_hidden_state.squeeze(0)}

class RobertaExtractor:
    """Text → RoBERTa/XLM-RoBERTa → pooling strategies."""
    def __init__(self, model_name: str = "FacebookAI/roberta-large",
                 device: str = "cuda",
                 output_mode: str = "pooled",
                 pooling_strategy: str = "cls"):  # "cls", "mean", "max"
        try:
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            ).to(device).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except ImportError:
            raise ImportError("transformers is required. pip install transformers")

        self.model_name = model_name
        self.device = device
        self.output_mode = output_mode
        self.pooling_strategy = pooling_strategy

    def fingerprint(self) -> str:
        return f"roberta:{self.model_name}:{self.pooling_strategy}"

    def _pool_embeddings(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to sequence embeddings."""
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling_strategy == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling_strategy == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            last_hidden_state[input_mask_expanded == 0] = -1e9
            return torch.max(last_hidden_state, 1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

    @torch.no_grad()
    def extract(self, text: Optional[str] = None, *args, **kwargs) -> Dict[str, torch.Tensor]:
        if text is None:
            if args and isinstance(args[0], str):
                text = args[0]
            else:
                text = ""

        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, return_dict=True)

        if self.output_mode == "pooled":
            embeddings = self._pool_embeddings(outputs.last_hidden_state, inputs["attention_mask"])
            return {"embedding": embeddings.squeeze(0)}

        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            valid_len = int(attention_mask.sum(dim=1).item())
            last_hidden_state = last_hidden_state[:, :valid_len, :]
        return {"embedding": last_hidden_state.squeeze(0)}

# ─────────────────────────────────────────────────────────────────────
# Factory: model names are read from config (or "off")
# ─────────────────────────────────────────────────────────────────────
def build_extractors_from_config(cfg) -> Dict[str, Any]:
    """
    cfg.video_extractor    = "openai/clip-vit-base-patch32" | "off"
    cfg.audio_extractor    = "laion/clap-htsat-fused"       | "off"
    cfg.text_extractor     = "openai/clip-vit-base-patch32" | "laion/clap-htsat-fused" | "off"
    cfg.behavior_extractor = "openai/clip-vit-base-patch32" | "laion/clap-htsat-fused" | "off"
    cfg.device             = "cuda" | "cpu" | "cuda:0"
    cfg.average_features   = "mean" | "mean_std" | "raw"
    """
    device = cfg.device

    # Map average_features to output mode:
    #   raw / mean_std → "seq"
    #   mean           → "pooled"
    af = cfg.average_features
    output_mode = "seq" if af in ("raw", "mean_std") else "pooled"

    ex: Dict[str, Any] = {}

    # video
    vid_model: str = cfg.video_extractor
    if isinstance(vid_model, str) and vid_model.lower() != "off":
        if "clip" in vid_model.lower():
            ex["face"] = ClipVideoExtractor(model_name=vid_model, device=device, output_mode=output_mode)
        else:
            raise ValueError(f"Video extractor '{vid_model}' is not supported (expected CLIP).")

    # audio
    aud_model: str = cfg.audio_extractor
    if isinstance(aud_model, str) and aud_model.lower() != "off":
        if "clap" in aud_model.lower():
            ex["audio"] = ClapAudioExtractor(model_name=aud_model, device=device, output_mode=output_mode)
        elif "wav2vec2" in aud_model.lower():
            ex["audio"] = HFAudioWav2Vec2(model_name=aud_model, device=device, output_mode=output_mode)
        else:
            raise ValueError(f"Audio extractor '{aud_model}' is not supported (expected CLAP).")

    # text
    txt_model: str = cfg.text_extractor
    if isinstance(txt_model, str) and txt_model.lower() != "off":
        if "clip" in txt_model.lower():
            ex["text"] = ClipTextExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        elif "clap" in txt_model.lower():
            ex["text"] = ClapTextExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        elif 'emotion_text_classifier' in txt_model.lower():
            ex['text'] = ETC_TextExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        elif 'roberta' in txt_model.lower():
            ex['text'] = RobertaExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        else:
            raise ValueError(f"Text extractor '{txt_model}' is not supported.")

    # behavior
    beh_model: str = cfg.behavior_extractor
    if isinstance(beh_model, str) and beh_model.lower() != "off":
        if "clip" in beh_model.lower():
            ex["behavior"] = ClipTextExtractor(model_name=beh_model, device=device, output_mode=output_mode)
        elif "clap" in beh_model.lower():
            ex["behavior"] = ClapTextExtractor(model_name=beh_model, device=device, output_mode=output_mode)
        elif 'emotion_text_classifier' in txt_model.lower():
            ex['behavior'] = ETC_TextExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        elif 'roberta' in txt_model.lower():
            ex['behavior'] = RobertaExtractor(model_name=txt_model, device=device, output_mode=output_mode)
        else:
            raise ValueError(f"Text extractor '{txt_model}' is not supported.")

    return ex


__all__ = [
    "ClipVideoExtractor", "ClipTextExtractor",
    "ClapAudioExtractor", "ClapTextExtractor",
    "HFAudioWav2Vec2",
    "ETC_TextExtractor", "RobertaExtractor"
    "build_extractors_from_config",
]
