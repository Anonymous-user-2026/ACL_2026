# coding: utf-8
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from .attention.crossmpt.Model_CrossMPT import (
    MultiHeadedAttention,
    PositionwiseFeedForward,
    Encoder,
    EncoderLayer,
)
from .layers import GraphAttentionLayer_V2

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────

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

class IdentityLayer(nn.Module):
    """Pass-through layer with the same call signature as graph layers."""
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x, *args, **kwargs):
        return x

# ────────────────────────────────────────────────────────────────────────────────
# Core blocks (original behavior preserved)
# ────────────────────────────────────────────────────────────────────────────────
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.proj(x)

class AdapterFusion(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.layernorm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        return self.layernorm(x + self.adapter(x))

class GuideBank(nn.Module):
    def __init__(self, out_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Parameter(torch.randn(out_dim, hidden_dim))
    def forward(self):
        return self.embeddings

class SemanticGuideBank(nn.Module):
    def __init__(self, class_names: List[str], hidden_dim: int, clip_model_name="openai/clip-vit-base-patch32", device="cuda"):
        super().__init__()
        self.class_names = class_names
        self.hidden_dim = hidden_dim
        self.device = _ensure_device(device)
        self.model = CLIPModel.from_pretrained(clip_model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(clip_model_name)
        with torch.no_grad():
            inputs = self.proc(text=[f"a photo of a {c}" for c in class_names], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embs = self.model.get_text_features(**inputs)
            if hidden_dim != 512:
                proj = nn.Linear(512, hidden_dim).to(self.device)
                text_embs = proj(text_embs)
            self.embeddings = nn.Parameter(text_embs)
    def forward(self):
        return self.embeddings

class DynamicAdjacencyLayer(nn.Module):
    def __init__(self, hidden_dim, temperature=1.0, learnable_temp=True):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=learnable_temp)
    def forward(self, h):
        sim = F.cosine_similarity(h.unsqueeze(2), h.unsqueeze(1), dim=-1)
        adj = torch.softmax(sim / self.temperature, dim=-1)
        return adj

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        weights = self.attention(x)
        return torch.sum(weights * x, dim=1)

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim=None, dropout=0.1, alpha=0.2):
        super().__init__()
        out_dim = out_dim or in_dim
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim
    def forward(self, h, adj):
        B, N, _ = h.size()
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        neg_inf = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, neg_inf)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, Wh)
        return h_prime

# ────────────────────────────────────────────────────────────────────────────────
# Ablation configuration
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class AblationCfg:
    use_graph: Optional[bool] = None         # None → True by default
    use_attention: Optional[bool] = None     # None → True by default
    use_guidebank: Optional[bool] = None     # None → True by default
    use_task_projectors: Optional[bool] = None
    disabled_modalities: Optional[List[str]] = None
    active_tasks: Optional[List[str]] = None

    @classmethod
    def from_obj(cls, o: Optional[object]) -> "AblationCfg":
        if o is None:
            return cls()
        if isinstance(o, dict):
            return cls(**{k: v for k, v in o.items() if k in cls.__annotations__})
        # object-like
        kw = {}
        for k in cls.__annotations__.keys():
            if hasattr(o, k):
                kw[k] = getattr(o, k)
        return cls(**kw)

# ────────────────────────────────────────────────────────────────────────────────
# Ablation-aware mixin shared by v1/v2/v3
# ────────────────────────────────────────────────────────────────────────────────
class _AblationMixin:
    def _apply_ablation_setup(self,
                              modality_input_dim: Dict[str, int],
                              hidden_dim: int,
                              num_heads: int,
                              out_dim: int,
                              emo_out_dim: int,
                              pkl_out_dim: int,
                              ah_out_dim: int,
                              device: str,
                              dropout: float,
                              ablation_cfg: Optional[AblationCfg]):
        self.hidden_dim = hidden_dim
        self.device = device
        self.out_dim = out_dim

        ab = AblationCfg.from_obj(ablation_cfg)
        self._use_attention = True if ab.use_attention is None else bool(ab.use_attention)
        use_graph_flag = True if ab.use_graph is None else bool(ab.use_graph)
        self.use_guidebank = True if ab.use_guidebank is None else bool(ab.use_guidebank)
        self.use_task_projectors = True if ab.use_task_projectors is None else bool(ab.use_task_projectors)

        # ── STRICT VALIDATION ─────────────
        # 1) validate tasks if provided
        allowed_tasks = {"emotion", "personality", "ah"}
        if ab.active_tasks is not None:
            unknown_tasks = set(ab.active_tasks) - allowed_tasks
            if unknown_tasks:
                raise KeyError(
                    f"[ablation.active_tasks] unknown: {sorted(unknown_tasks)}; "
                    f"known: {sorted(allowed_tasks)}"
                )
            if len(ab.active_tasks) == 0:
                raise ValueError("[ablation.active_tasks] cannot be empty — specify at least one task")

        # 2) validate disabled modalities against provided input dims
        allowed_modalities = set(modality_input_dim.keys())
        disabled = set(ab.disabled_modalities or [])
        unknown_modalities = disabled - allowed_modalities
        if unknown_modalities:
            raise KeyError(
                f"[ablation.disabled_modalities] unknown modalities: {sorted(unknown_modalities)}; "
                f"known: {sorted(allowed_modalities)}"
            )

        # tasks / modalities (tasks only from ablation or default set)
        self.active_tasks = tuple(ab.active_tasks) if ab.active_tasks else ("emotion", "personality", "ah")
        self.modalities = {m: d for m, d in modality_input_dim.items() if m not in disabled}
        if not self.modalities:
            raise ValueError("All modalities are disabled — nothing to fuse.")

        # modality projectors
        self.modality_projectors = nn.ModuleDict({
            mod: nn.Sequential(
                Projector(in_dim, hidden_dim, dropout=dropout),
                AdapterFusion(hidden_dim, dropout=dropout),
            ) for mod, in_dim in self.modalities.items()
        })

        self.graph_attns = nn.ModuleDict()
        self.prediction_projectors = nn.ModuleDict()
        self.cross_attns = nn.ModuleDict()
        self.guide_banks = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.heads = nn.ModuleDict()

        task_dims = {"emotion": emo_out_dim, "personality": pkl_out_dim, "ah": ah_out_dim}

        # select graph layer variant per model
        GraphCls = self._select_graph_layer(use_graph_flag)

        for task in self.active_tasks:
            out_dim_t = task_dims[task]

            if self.use_task_projectors:
                self.predictors[task] = nn.Sequential(
                    Projector(hidden_dim, out_dim_t, dropout=dropout),
                    AdapterFusion(out_dim_t, dropout=dropout),
                )
                self.prediction_projectors[task] = nn.Sequential(
                    Projector(out_dim_t, hidden_dim, dropout=dropout),
                    AdapterFusion(hidden_dim, dropout=dropout),
                )

            # self.graph_attns[task] = GraphCls(hidden_dim, dropout=dropout)

            self.graph_attns[task] = (
                GraphCls(hidden_dim, dropout=dropout) if self.use_task_projectors
                else IdentityLayer()
            )

            # cross-attention holder (implementation picked by subclass)
            self._init_cross_attention_for_task(task, hidden_dim, num_heads, dropout, use_attention=self._use_attention)

            # guidebanks & heads
            self.guide_banks[task] = GuideBank(out_dim_t, hidden_dim)
            if task == "personality":
                self.heads[task] = nn.Sequential(
                    Projector(hidden_dim, self.out_dim, dropout=dropout),
                    nn.Linear(self.out_dim, out_dim_t),
                    nn.Sigmoid(),
                )
            else:
                self.heads[task] = nn.Sequential(
                    Projector(hidden_dim, self.out_dim, dropout=dropout),
                    nn.Linear(self.out_dim, out_dim_t),
                )

        # graph over modality features
        self.graph_attns["features"] = GraphCls(hidden_dim, dropout=dropout)

    @staticmethod
    def _temporal_pool(x: torch.Tensor | None) -> torch.Tensor | None:
        if x is None:
            return None
        if x.dim() == 3:
            return x.mean(dim=1)
        if x.dim() == 2:
            return x
        raise ValueError(f"Unexpected feature shape {tuple(x.shape)}")

    # Hooks for subclasses
    def _select_graph_layer(self, use_graph: bool):
        raise NotImplementedError

    def _init_cross_attention_for_task(self, task: str, hidden_dim: int, num_heads: int, dropout: float, *, use_attention: bool):
        raise NotImplementedError

    def _cross_attention_forward(self, task: str, ctx_preds: torch.Tensor, ctx_mods: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

# ────────────────────────────────────────────────────────────────────────────────
# v1: original GraphAttentionLayer + PyTorch MultiheadAttention
# ────────────────────────────────────────────────────────────────────────────────
class MultiModalFusionModel_v1(_AblationMixin, nn.Module):
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {"face": 512, "audio": 512, "text": 768, "behavior": 512},
        hidden_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 256,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        device: str = "cpu",
        dropout: float = 0.1,
        ablation_cfg: Optional[AblationCfg] = None,
    ):
        super().__init__()
        self._apply_ablation_setup(
            modality_input_dim, hidden_dim, num_heads, out_dim,
            emo_out_dim, pkl_out_dim, ah_out_dim,
            device, dropout, ablation_cfg,
        )

    def _select_graph_layer(self, use_graph: bool):
        return GraphAttentionLayer if use_graph else IdentityLayer

    def _init_cross_attention_for_task(self, task, hidden_dim, num_heads, dropout, *, use_attention: bool):
        self.cross_attns[task] = None if not use_attention else nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )

    def _cross_attention_forward(self, task, ctx_preds, ctx_mods):
        if not self._use_attention or self.cross_attns[task] is None:
            return ctx_preds.mean(dim=1)
        task_repr, _ = self.cross_attns[task](ctx_preds, ctx_mods, ctx_mods)  # [B,N,H]
        return task_repr.mean(dim=1)

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []
        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)
        if not x_mods:
            raise ValueError("No valid modalities provided")

        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)  # [B,N,H]
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            if self.use_task_projectors:
                task_logits_per_mod = self.predictors[task](x_stack)                # [B,N,C]
                preds_stack = self.prediction_projectors[task](task_logits_per_mod) # [B,N,H]
                adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
                ctx_preds = self.graph_attns[task](preds_stack, adj_t)              # [B,N,H]
            else:
                ctx_preds = ctx_mods
                adj_t = torch.ones(B, ctx_preds.size(1), ctx_preds.size(1), device=self.device)
                ctx_preds = self.graph_attns[task](ctx_preds, adj_t)              # [B,N,H]

            task_repr = self._cross_attention_forward(task, ctx_preds, ctx_mods)
            logits = self.heads[task](task_repr)

            if self.use_guidebank:
                guides = self.guide_banks[task]()
                sim = F.cosine_similarity(task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1)
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits
        return outputs

# ────────────────────────────────────────────────────────────────────────────────
# v2: GraphAttentionLayer_V2 + PyTorch MultiheadAttention
# ────────────────────────────────────────────────────────────────────────────────
class MultiModalFusionModel_v2(MultiModalFusionModel_v1):
    def _select_graph_layer(self, use_graph: bool):
        return GraphAttentionLayer_V2 if use_graph else IdentityLayer

# ────────────────────────────────────────────────────────────────────────────────
# v3: GraphAttentionLayer_V2 + CrossMPT attention
# ────────────────────────────────────────────────────────────────────────────────
class MultiModalFusionModel_v3(_AblationMixin, nn.Module):
    def __init__(
        self,
        modality_input_dim: Dict[str, int] = {"face": 512, "audio": 512, "text": 768, "behavior": 512},
        hidden_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 256,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        device: str = "cpu",
        dropout: float = 0.1,
        ablation_cfg: Optional[AblationCfg] = None,
    ):
        super().__init__()
        self._apply_ablation_setup(
            modality_input_dim, hidden_dim, num_heads, out_dim,
            emo_out_dim, pkl_out_dim, ah_out_dim,
            device, dropout, ablation_cfg,
        )

    def _select_graph_layer(self, use_graph: bool):
        return GraphAttentionLayer_V2 if use_graph else IdentityLayer

    def _init_cross_attention_for_task(self, task, hidden_dim, num_heads, dropout, *, use_attention: bool):
        if not use_attention:
            self.cross_attns[task] = None
            return
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, hidden_dim)
        ff = PositionwiseFeedForward(hidden_dim, hidden_dim * 4, dropout)
        self.cross_attns[task] = Encoder(EncoderLayer(hidden_dim, c(attn), c(ff), dropout), 1)

    def _cross_attention_forward(self, task, ctx_preds, ctx_mods):
        if not self._use_attention or self.cross_attns[task] is None:
            return ctx_preds.mean(dim=1)
        emb1, emb2 = self.cross_attns[task](ctx_preds, ctx_mods, None, None)
        task_repr = torch.cat([emb1, emb2], dim=1).mean(dim=1)
        return task_repr

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]):
        feats = batch["features"]
        x_mods: Dict[str, torch.Tensor] = {}
        valid = []
        for mod, tensor in feats.items():
            if mod not in self.modality_projectors or tensor is None:
                continue
            x = self._temporal_pool(tensor.to(self.device))
            x_mods[mod] = self.modality_projectors[mod](x)
            valid.append(mod)
        if not x_mods:
            raise ValueError("No valid modalities provided")

        x_stack = torch.stack([x_mods[m] for m in valid], dim=1)
        B, N, H = x_stack.shape
        adj = torch.ones(B, N, N, device=self.device)
        ctx_mods = self.graph_attns["features"](x_stack, adj)

        outputs = {"emotion_logits": None, "personality_scores": None, "ah_logits": None}

        for task in self.active_tasks:
            if self.use_task_projectors:
                task_logits_per_mod = self.predictors[task](x_stack)
                preds_stack = self.prediction_projectors[task](task_logits_per_mod)
                adj_t = torch.ones(B, preds_stack.size(1), preds_stack.size(1), device=self.device)
                ctx_preds = self.graph_attns[task](preds_stack, adj_t)
            else:
                ctx_preds = ctx_mods
                adj_t = torch.ones(B, ctx_preds.size(1), ctx_preds.size(1), device=self.device)
                ctx_preds = self.graph_attns[task](ctx_preds, adj_t)              # [B,N,H]

            task_repr = self._cross_attention_forward(task, ctx_preds, ctx_mods)
            logits = self.heads[task](task_repr)

            if self.use_guidebank:
                guides = self.guide_banks[task]()
                sim = F.cosine_similarity(task_repr.unsqueeze(1), guides.unsqueeze(0), dim=-1)
                if task == "personality":
                    logits = (logits + torch.sigmoid(sim)) / 2.0
                else:
                    logits = (logits + sim) / 2.0

            if task == "personality":
                outputs["personality_scores"] = logits
            elif task == "emotion":
                outputs["emotion_logits"] = logits
            elif task == "ah":
                outputs["ah_logits"] = logits
        return outputs
