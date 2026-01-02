# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class BellLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_p = torch.pow((y - p), 2)
        y_p_div = -1.0 * torch.div(y_p, 162.0)
        exp_y_p = torch.exp(y_p_div)
        loss = 300 * (1.0 - exp_y_p)
        return torch.mean(loss)


class LogCosh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = torch.log(torch.cosh(p - y))
        return torch.mean(loss)


class RMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(p, y))


class GL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.lam = lam
        self.eps = eps
        self.sigma = sigma

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gl = self.eps / (self.lam ** 2) * (1 - torch.exp(-1 * ((y - p) ** 2) / (self.sigma ** 2)))
        return gl.mean()


class RMBell(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y)


class RMLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.logcosh(p, y)


class RMGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.gl(p, y)


class RMBellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.logcosh(p, y)


class RMBellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.rmse = RMSE()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rmse(p, y) + self.bell(p, y) + self.gl(p, y)


class BellLCosh(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y)


class BellGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.bell = BellLoss()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.gl(p, y)


class BellLCoshGL(nn.Module):
    def __init__(self):
        super().__init__()
        self.bell = BellLoss()
        self.logcosh = LogCosh()
        self.gl = GL()

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.bell(p, y) + self.logcosh(p, y) + self.gl(p, y)


class LogCoshGL(nn.Module):
    def __init__(self, lam=1.0, eps=600, sigma=8):
        super().__init__()
        self.logcosh = LogCosh()
        self.gl = GL(lam=lam, eps=eps, sigma=sigma)

    def forward(self, p: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.logcosh(p, y) + self.gl(p, y)


class MAELoss(nn.Module):
    """Mean Absolute Error"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(x - y))


class MSELoss(nn.Module):
    """Mean Squared Error"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.pow(x - y, 2))


class CCCLoss(nn.Module):
    """Lin's Concordance Correlation Coefficient.
    https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Returns 1 - CCC."""
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps
        )
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,
        emo_weights=None,
        ah_weights=None,
        personality_loss_type: str = "ccc",
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality
        self.weight_ah = weight_ah

        # Emotion: CE
        self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)

        # AH: 2-class CE
        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)

        # Personality: choose by name
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Unknown personality_loss_type: {personality_loss_type}. Available: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Emotion (classification)
        if 'emotion_logits' in outputs and 'emotion' in labels:
            true_emotion = labels['emotion']              # (B,)
            pred_emotion = outputs['emotion_logits']      # (B, C_emo)
            loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Personality (regression over 5 traits)
        if 'personality_scores' in outputs and 'personality' in labels:
            true_personality = labels['personality']          # (B, 5)
            pred_personality = outputs['personality_scores']  # (B, 5)

            if self.personality_loss_type == "ccc":
                loss_per = 0.0
                for i in range(5):
                    loss_per += self.personality_loss(true_personality[:, i], pred_personality[:, i])
                loss += loss_per * self.weight_personality
            else:
                loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        # AH (binary)
        if 'ah_logits' in outputs and 'ah' in labels:
            true_ah = labels['ah']           # (B,) int64 with classes 0/1
            pred_ah = outputs['ah_logits']   # (B, 2)
            loss += self.weight_ah * self.ah_loss(pred_ah, true_ah)

        return loss


def binarize_with_nan(x, threshold=0.5):
    """Binarize values > threshold while preserving NaNs."""
    nan_mask = torch.isnan(x)
    binary = torch.zeros_like(x)
    binary[x > threshold] = 1.0
    binary[nan_mask] = float('nan')
    return binary


class MultiTaskLossWithNaN(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,
        emo_weights=None,
        ah_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = 'BCE',  # CE/BCE
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality
        self.weight_ah = weight_ah

        # Emotion: CE or BCE
        if emotion_loss_type == 'CE':
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        elif emotion_loss_type == 'BCE':
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        # AH: CE (2-class)
        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)

        # Personality: choose by name
        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Unknown personality_loss_type: {personality_loss_type}. Available: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Emotion branch (with mask)
        emo_mask = labels.get('valid_emo', None)
        pred_emotion_all = outputs.get('emotion_logits')
        if pred_emotion_all is not None:
            if emo_mask is None:
                true_emotion = labels['emotion']
                pred_emotion = pred_emotion_all
                valid_any = True
            else:
                valid_any = emo_mask.any()
                if valid_any:
                    true_emotion = labels['emotion'][emo_mask]
                    pred_emotion = pred_emotion_all[emo_mask]

            if pred_emotion_all is not None and (emo_mask is None or valid_any):
                if self.emotion_loss_type == 'BCE':
                    true_emotion = binarize_with_nan(true_emotion, threshold=0)
                loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # Personality branch (mask/NaN per trait)
        per_mask = labels.get('valid_per', None)
        pred_personality_all = outputs.get('personality_scores')
        if pred_personality_all is not None:
            if per_mask is None:
                true_personality = labels['personality']
                pred_personality = pred_personality_all
                per_valid_any = True
            else:
                per_valid_any = per_mask.any()
                if per_valid_any:
                    true_personality = labels['personality'][per_mask]
                    pred_personality = pred_personality_all[per_mask]

            if pred_personality_all is not None and (per_mask is None or per_valid_any):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0
                    valid_traits = 0
                    for i in range(5):
                        trait_mask = ~torch.isnan(true_personality[:, i])
                        if trait_mask.any():
                            loss_per += self.personality_loss(
                                true_personality[trait_mask, i],
                                pred_personality[trait_mask, i]
                            )
                            valid_traits += 1
                    if valid_traits > 0:
                        loss += (loss_per / valid_traits) * self.weight_personality
                else:
                    loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        # AH branch (supports valid_ah mask if present)
        ah_logits_all = outputs.get('ah_logits')  # (B, 2)
        if ah_logits_all is not None:
            ah_mask = labels.get('valid_ah', None)
            if ah_mask is None:
                # If there are NaNs in y and no mask, CE would fail. Fallback to masking NaNs here.
                true_ah = labels['ah']
                if true_ah.dtype != torch.long:
                    nan_mask = torch.isnan(true_ah) if true_ah.dtype.is_floating_point else torch.zeros_like(true_ah, dtype=torch.bool)
                    if nan_mask.any():
                        ah_mask = ~nan_mask
                        true_ah = true_ah[ah_mask]
                        pred_ah = ah_logits_all[ah_mask]
                    else:
                        pred_ah = ah_logits_all
                    true_ah = true_ah.long()
                    ah_valid_any = true_ah.numel() > 0
                else:
                    pred_ah = ah_logits_all
                    ah_valid_any = True
            else:
                ah_valid_any = ah_mask.any()
                if ah_valid_any:
                    true_ah = labels['ah'][ah_mask]
                    if true_ah.dtype != torch.long:
                        true_ah = true_ah.long()
                    pred_ah = ah_logits_all[ah_mask]

            if ah_valid_any:
                loss += self.weight_ah * self.ah_loss(pred_ah, true_ah)

        if not isinstance(loss, torch.Tensor):
            device = (
                (outputs.get("emotion_logits") or outputs.get("personality_scores") or outputs.get("ah_logits")).device
                if (outputs.get("emotion_logits") is not None
                    or outputs.get("personality_scores") is not None
                    or outputs.get("ah_logits") is not None)
                else torch.device("cpu")
            )
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss


class MultiTaskLossWithNaN_v2(nn.Module):
    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,
        emo_weights=None,
        ah_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = 'BCE',
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
        # SSL weights and thresholds
        ssl_weight_emotion: float = 0.0,
        ssl_weight_personality: float = 0.0,
        ssl_weight_ah: float = 0.0,
        ssl_confidence_threshold_emo_ah: float = 0.80,
        ssl_confidence_threshold_pt: float = 0.60,
    ):
        super().__init__()
        self.weight_emotion = weight_emotion
        self.weight_personality = weight_personality
        self.weight_ah = weight_ah

        self.ssl_weight_emotion = ssl_weight_emotion or 0.0
        self.ssl_weight_personality = ssl_weight_personality or 0.0
        self.ssl_weight_ah = ssl_weight_ah or 0.0
        self.ssl_confidence_threshold_emo_ah = ssl_confidence_threshold_emo_ah or 0.80
        self.ssl_confidence_threshold_pt = ssl_confidence_threshold_pt or 0.60

        if emotion_loss_type == 'CE':
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        elif emotion_loss_type == 'BCE':
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
            self.emotion_loss_type = emotion_loss_type
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)

        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(),
            "mse": MSELoss(),
            "bell": BellLoss(),
            "logcosh": LogCosh(),
            "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(),
            "rmse_bell": RMBell(),
            "rmse_logcosh": RMLCosh(),
            "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(),
            "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(),
            "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(),
            "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(
                f"Unknown personality_loss_type: {personality_loss_type}. Available: {list(loss_types.keys())}"
            )
        self.personality_loss = loss_types[personality_loss_type]
        self.personality_loss_type = personality_loss_type

    def forward(self, outputs, labels):
        loss = 0.0

        # Emotion branch (with mask)
        emo_mask = labels.get('valid_emo', None)
        pred_emotion_all = outputs.get('emotion_logits')
        if pred_emotion_all is not None:
            if emo_mask is None:
                true_emotion = labels['emotion']
                pred_emotion = pred_emotion_all
                valid_any = True
            else:
                valid_any = emo_mask.any()
                if valid_any:
                    true_emotion = labels['emotion'][emo_mask]
                    pred_emotion = pred_emotion_all[emo_mask]

            if pred_emotion_all is not None and (emo_mask is None or valid_any):
                if self.emotion_loss_type == 'BCE':
                    true_emotion = binarize_with_nan(true_emotion, threshold=0)
                loss += self.weight_emotion * self.emotion_loss(pred_emotion, true_emotion)

        # SSL: emotion pseudo-labels on unlabeled samples
        if self.ssl_weight_emotion > 0.0 and pred_emotion_all is not None:
            emo_mask = labels.get('valid_emo', None)
            if emo_mask is not None:
                unlabeled_mask = ~emo_mask
                if unlabeled_mask.any():
                    pred_emotion_unlabeled = pred_emotion_all[unlabeled_mask]
                    if self.emotion_loss_type == 'BCE':
                        probs = torch.sigmoid(pred_emotion_unlabeled)
                        confidence, pseudo_labels = torch.max(probs, dim=1)
                        mask_confident = confidence > self.ssl_confidence_threshold_emo_ah
                    elif self.emotion_loss_type == 'CE':
                        probs = torch.softmax(pred_emotion_unlabeled, dim=1)
                        confidence, pseudo_labels = torch.max(probs, dim=1)
                        mask_confident = confidence > self.ssl_confidence_threshold_emo_ah
                    else:
                        mask_confident = torch.zeros(unlabeled_mask.sum(), dtype=torch.bool, device=pred_emotion_all.device)

                    if mask_confident.any():
                        pred_emotion_confident = pred_emotion_unlabeled[mask_confident]
                        pseudo_labels_confident = pseudo_labels[mask_confident]
                        if self.emotion_loss_type == 'BCE':
                            pseudo_labels_confident = pseudo_labels_confident.float()
                        loss += self.ssl_weight_emotion * self.emotion_loss(pred_emotion_confident, pseudo_labels_confident)

        # Personality branch (mask/NaN per trait)
        per_mask = labels.get('valid_per', None)
        pred_personality_all = outputs.get('personality_scores')
        if pred_personality_all is not None:
            if per_mask is None:
                true_personality = labels['personality']
                pred_personality = pred_personality_all
                per_valid_any = True
            else:
                per_valid_any = per_mask.any()
                if per_valid_any:
                    true_personality = labels['personality'][per_mask]
                    pred_personality = pred_personality_all[per_mask]

            if pred_personality_all is not None and (per_mask is None or per_valid_any):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0
                    valid_traits = 0
                    for i in range(5):
                        trait_mask = ~torch.isnan(true_personality[:, i])
                        if trait_mask.any():
                            loss_per += self.personality_loss(
                                true_personality[trait_mask, i],
                                pred_personality[trait_mask, i]
                            )
                            valid_traits += 1
                    if valid_traits > 0:
                        loss += (loss_per / valid_traits) * self.weight_personality
                else:
                    loss += self.weight_personality * self.personality_loss(true_personality, pred_personality)

        # SSL: personality pseudo-labels via BCE on confident values
        if self.ssl_weight_personality > 0.0 and pred_personality_all is not None:
            per_mask = labels.get('valid_per', None)
            if per_mask is not None:
                unlabeled_mask = ~per_mask
                if unlabeled_mask.any():
                    pred_per_unlabeled = pred_personality_all[unlabeled_mask]  # (U, 5)
                    pred_per_unlabeled = torch.clamp(pred_per_unlabeled, 0.0, 1.0)
                    pseudo_labels = (pred_per_unlabeled > 0.5).float()  # (U, 5)
                    confidence_mask = (
                        (pred_per_unlabeled > self.ssl_confidence_threshold_pt) |
                        (pred_per_unlabeled < (1 - self.ssl_confidence_threshold_pt))
                    )
                    if confidence_mask.any():
                        bce_loss_per_element = F.binary_cross_entropy(
                            pred_per_unlabeled,
                            pseudo_labels,
                            reduction='none'
                        )
                        weighted_loss = (bce_loss_per_element * confidence_mask.float()).sum()
                        total_confident = confidence_mask.sum().float()
                        if total_confident > 0:
                            ssl_loss = weighted_loss / total_confident
                            loss += self.ssl_weight_personality * ssl_loss

        # AH branch (supports valid_ah mask if present)
        ah_logits_all = outputs.get('ah_logits')  # (B, 2)
        if ah_logits_all is not None:
            ah_mask = labels.get('valid_ah', None)
            if ah_mask is None:
                # If there are NaNs in y and no mask, CE would fail. Fallback to masking NaNs here.
                true_ah = labels['ah']
                if true_ah.dtype != torch.long:
                    nan_mask = torch.isnan(true_ah) if true_ah.dtype.is_floating_point else torch.zeros_like(true_ah, dtype=torch.bool)
                    if nan_mask.any():
                        ah_mask = ~nan_mask
                        true_ah = true_ah[ah_mask]
                        pred_ah = ah_logits_all[ah_mask]
                    else:
                        pred_ah = ah_logits_all
                    true_ah = true_ah.long()
                    ah_valid_any = true_ah.numel() > 0
                else:
                    pred_ah = ah_logits_all
                    ah_valid_any = True
            else:
                ah_valid_any = ah_mask.any()
                if ah_valid_any:
                    true_ah = labels['ah'][ah_mask]
                    if true_ah.dtype != torch.long:
                        true_ah = true_ah.long()
                    pred_ah = ah_logits_all[ah_mask]

            if ah_valid_any:
                loss += self.weight_ah * self.ah_loss(pred_ah, true_ah)

        # SSL: AH pseudo-labels on unlabeled samples
        if self.ssl_weight_ah > 0.0 and ah_logits_all is not None:
            ah_mask = labels.get('valid_ah', None)
            if ah_mask is not None:
                unlabeled_mask = ~ah_mask
                if unlabeled_mask.any():
                    pred_ah_unlabeled = ah_logits_all[unlabeled_mask]  # (U, 2)
                    probs = torch.softmax(pred_ah_unlabeled, dim=1)
                    confidence, pseudo_labels = torch.max(probs, dim=1)
                    mask_confident = confidence > self.ssl_confidence_threshold_emo_ah
                    if mask_confident.any():
                        pred_ah_confident = pred_ah_unlabeled[mask_confident]
                        pseudo_labels_confident = pseudo_labels[mask_confident]
                        loss += self.ssl_weight_ah * self.ah_loss(pred_ah_confident, pseudo_labels_confident)

        if not isinstance(loss, torch.Tensor):
            device = (
                (outputs.get("emotion_logits") or outputs.get("personality_scores") or outputs.get("ah_logits")).device
                if (outputs.get("emotion_logits") is not None
                    or outputs.get("personality_scores") is not None
                    or outputs.get("ah_logits") is not None)
                else torch.device("cpu")
            )
            loss = torch.tensor(0.0, requires_grad=True, device=device)

        return loss


def _binarize_with_nan(x: torch.Tensor, threshold=0.0):
    mask = ~torch.isnan(x)
    out = torch.zeros_like(x)
    out[mask] = (x[mask] > threshold).float()
    return out


class MultiTaskLossWithNaN_v3(nn.Module):
    SUP_KEYS = ["emo_sup", "per_sup", "ah_sup"]
    SSL_KEYS = ["emo_ssl", "per_ssl", "ah_ssl"]

    def __init__(
        self,
        weight_emotion: float = 1.0,
        weight_personality: float = 1.0,
        weight_ah: float = 1.0,
        emo_weights=None,
        ah_weights=None,
        personality_loss_type: str = "ccc",
        emotion_loss_type: str = "BCE",
        eps: float = 1e-8,
        lam_gl: float = 1.0,
        eps_gl: float = 600,
        sigma_gl: float = 8,
        # SSL thresholds
        ssl_confidence_threshold_emo_ah: float = 0.80,
        ssl_confidence_threshold_pt: float = 0.60,
        # GradNorm hyperparameters
        alpha_sup: float = 1.25,
        w_lr_sup: float = 0.025,
        alpha_ssl: float = 0.75,
        w_lr_ssl: float = 0.001,
        lambda_ssl: float = 0.2,
        w_floor: float = 1e-3
    ):
        super().__init__()
        self.eps = eps
        self.ssl_confidence_threshold_emo_ah = float(ssl_confidence_threshold_emo_ah)
        self.ssl_confidence_threshold_pt     = float(ssl_confidence_threshold_pt)

        self.emotion_loss_type = emotion_loss_type
        if emotion_loss_type == "CE":
            self.emotion_loss = nn.CrossEntropyLoss(weight=emo_weights)
        elif emotion_loss_type == "BCE":
            self.emotion_loss = nn.BCEWithLogitsLoss(weight=emo_weights)
        else:
            raise ValueError(f"Unknown emotion_loss_type: {emotion_loss_type}")

        self.ah_loss = nn.CrossEntropyLoss(weight=ah_weights)

        loss_types = {
            "ccc": CCCLoss(eps=eps),
            "mae": MAELoss(), "mse": MSELoss(),
            "bell": BellLoss(), "logcosh": LogCosh(), "gl": GL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse": RMSE(), "rmse_bell": RMBell(), "rmse_logcosh": RMLCosh(), "rmse_gl": RMGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "rmse_bell_logcosh": RMBellLCosh(), "rmse_bell_gl": RMBellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh": BellLCosh(), "bell_gl": BellGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
            "bell_logcosh_gl": BellLCoshGL(), "logcosh_gl": LogCoshGL(lam=lam_gl, eps=eps_gl, sigma=sigma_gl),
        }
        if personality_loss_type not in loss_types:
            raise ValueError(f"Unknown personality_loss_type: {personality_loss_type}")
        self.personality_loss_type = personality_loss_type
        self.personality_loss = loss_types[personality_loss_type]

        self.alpha_sup = float(alpha_sup)
        self.w_lr_sup  = float(w_lr_sup)
        self.alpha_ssl = float(alpha_ssl)
        self.w_lr_ssl  = float(w_lr_ssl)
        self.lambda_ssl = float(lambda_ssl)
        self.w_floor = float(w_floor)
        self.budget_sup = 3.0
        self.budget_ssl = 3.0 * self.lambda_ssl

        self.weight_sup = nn.ParameterDict({
            "emo_sup": nn.Parameter(torch.tensor(float(weight_emotion))),
            "per_sup": nn.Parameter(torch.tensor(float(weight_personality))),
            "ah_sup":  nn.Parameter(torch.tensor(float(weight_ah))),
        })

        self.weight_ssl = nn.ParameterDict({
            "emo_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
            "per_ssl": nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
            "ah_ssl":  nn.Parameter(torch.tensor(self.lambda_ssl, dtype=torch.float32)),
        })

        self._normalize(self.weight_sup, self.SUP_KEYS, self.budget_sup)
        self._normalize(self.weight_ssl, self.SSL_KEYS, self.budget_ssl)

        self.init_sup = {}
        self.init_ssl = {}

    @staticmethod
    def _shared_params_from_model(model: nn.Module):
        return [p for name, p in model.named_parameters()
                if not ("emotion" in name or "personality" in name or "ah" in name)]

    @staticmethod
    def _to_onehot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
        return F.one_hot(indices, num_classes=num_classes).float()

    @staticmethod
    def _mean_abs_norm(grads):
        g = [t.detach().flatten() for t in grads if t is not None]
        if not g:
            return None
        return torch.norm(torch.cat(g), p=2)

    @staticmethod
    def _safe_detach(x: torch.Tensor) -> torch.Tensor:
        return x.detach() if isinstance(x, torch.Tensor) else torch.tensor(float(x))

    def _normalize(self, pdict: nn.ParameterDict, keys, target_sum: float):
        with torch.no_grad():
            s = sum(pdict[k] for k in keys)
            sval = s.detach().clamp_min(1e-8)
            for k in keys:
                pdict[k].data = target_sum * (pdict[k].data / sval)

    def _collect_components(self, outputs, labels):
        comps = {}

        # Supervised EMO
        pred_e = outputs.get("emotion_logits", None)
        emo_mask = labels.get("valid_emo", None)
        if pred_e is not None:
            if emo_mask is None:
                true_e = labels["emotion"]
                pred_e_sup = pred_e
                any_sup = True
            else:
                any_sup = emo_mask.any()
                if any_sup:
                    true_e = labels["emotion"][emo_mask]
                    pred_e_sup = pred_e[emo_mask]
            if pred_e is not None and (emo_mask is None or any_sup):
                if self.emotion_loss_type == "BCE":
                    true_e = _binarize_with_nan(true_e, threshold=0.0)
                    comps["emo_sup"] = self.emotion_loss(pred_e_sup, true_e)
                else:  # CE
                    target_e = (torch.argmax(true_e, dim=1) if true_e.dim() > 1 else true_e.long())
                    comps["emo_sup"] = self.emotion_loss(pred_e_sup, target_e)

        # SSL EMO
        if pred_e is not None and emo_mask is not None:
            unlabeled = ~emo_mask
            if unlabeled.any():
                pred_u = pred_e[unlabeled]
                probs = torch.sigmoid(pred_u) if self.emotion_loss_type == "BCE" else torch.softmax(pred_u, dim=1)
                conf, pseudo = torch.max(probs, dim=1)
                c_mask = conf > self.ssl_confidence_threshold_emo_ah
                if c_mask.any():
                    pred_c = pred_u[c_mask]
                    if self.emotion_loss_type == "BCE":
                        num_c = pred_c.size(1)
                        pseudo_c = self._to_onehot(pseudo[c_mask], num_c)
                        comps["emo_ssl"] = self.emotion_loss(pred_c, pseudo_c)
                    else:
                        comps["emo_ssl"] = self.emotion_loss(pred_c, pseudo[c_mask])

        # Supervised PER
        pred_p = outputs.get("personality_scores", None)
        per_mask = labels.get("valid_per", None)
        if pred_p is not None:
            if per_mask is None:
                tp = labels["personality"]; pp = pred_p
                any_sup = True
            else:
                any_sup = per_mask.any()
                if any_sup:
                    tp = labels["personality"][per_mask]; pp = pred_p[per_mask]
            if pred_p is not None and (per_mask is None or any_sup):
                if self.personality_loss_type == "ccc":
                    loss_per = 0.0; valid_traits = 0
                    for i in range(tp.shape[1]):
                        tmask = ~torch.isnan(tp[:, i])
                        if tmask.any():
                            loss_per = loss_per + self.personality_loss(tp[tmask, i], pp[tmask, i])
                            valid_traits += 1
                    if valid_traits > 0:
                        comps["per_sup"] = loss_per / valid_traits
                else:
                    comps["per_sup"] = self.personality_loss(tp, pp)

        # SSL PER
        if pred_p is not None and per_mask is not None:
            unlabeled = ~per_mask
            if unlabeled.any():
                pu = torch.clamp(pred_p[unlabeled], 0.0, 1.0)
                pseudo = (pu > 0.5).float()
                c_mask = ((pu > self.ssl_confidence_threshold_pt) | (pu < (1.0 - self.ssl_confidence_threshold_pt)))
                if c_mask.any():
                    bce_per_elem = F.binary_cross_entropy(pu, pseudo, reduction="none")
                    weighted = (bce_per_elem * c_mask.float()).sum()
                    tot = c_mask.sum().float()
                    if tot > 0:
                        comps["per_ssl"] = weighted / tot

        # Supervised AH
        pred_a = outputs.get("ah_logits", None)
        ah_mask = labels.get("valid_ah", None)
        if pred_a is not None:
            if ah_mask is None:
                y = labels["ah"]
                if y.dtype != torch.long:
                    nan_mask = torch.isnan(y) if y.dtype.is_floating_point else torch.zeros_like(y, dtype=torch.bool)
                    if nan_mask.any():
                        ah_mask = ~nan_mask
                        y = y[ah_mask]; pred_ah_sup = pred_a[ah_mask]
                    else:
                        pred_ah_sup = pred_a
                    y = y.long()
                    any_sup = (y.numel() > 0)
                else:
                    pred_ah_sup = pred_a
                    any_sup = True
            else:
                any_sup = ah_mask.any()
                if any_sup:
                    y = labels["ah"][ah_mask]
                    if y.dtype != torch.long: y = y.long()
                    pred_ah_sup = pred_a[ah_mask]
            if any_sup:
                comps["ah_sup"] = self.ah_loss(pred_ah_sup, y)

        # SSL AH
        if pred_a is not None and ah_mask is not None:
            unlabeled = ~ah_mask
            if unlabeled.any():
                pu = pred_a[unlabeled]
                probs = torch.softmax(pu, dim=1)
                conf, pseudo = torch.max(probs, dim=1)
                c_mask = conf > self.ssl_confidence_threshold_emo_ah
                if c_mask.any():
                    comps["ah_ssl"] = self.ah_loss(pu[c_mask], pseudo[c_mask])

        return comps

    def _gradnorm_update_wallet(self, comps, keys, init_dict, weight_pdict, alpha, w_lr, budget, shared_params):
        for k in keys:
            Li = comps.get(k, None)
            if (Li is not None) and (k not in init_dict) and torch.isfinite(Li).all():
                init_dict[k] = Li.detach().clamp_min(1e-8)

        active = [k for k in keys if (k in comps) and (k in init_dict)]
        if not active:
            return

        G_list, r_list, w_list = [], [], []
        for k in active:
            Li = comps[k]
            grads = torch.autograd.grad(Li, shared_params, retain_graph=True, allow_unused=True)
            grad_norm = self._mean_abs_norm(grads) or torch.tensor(0.0, device=Li.device)
            wk = weight_pdict[k]
            G_list.append(wk * grad_norm)
            r_list.append((Li.detach().clamp_min(1e-8) / init_dict[k]))
            w_list.append(wk)

        G_stack = torch.stack(G_list); r_stack = torch.stack(r_list)
        G_avg, r_avg = G_stack.mean(), r_stack.mean()
        gn_loss = 0.0
        for i, _ in enumerate(active):
            target = (G_avg * ((r_stack[i] / r_avg) ** alpha)).detach()
            gn_loss = gn_loss + torch.abs(G_stack[i] - target)

        grads_w = torch.autograd.grad(gn_loss, w_list, retain_graph=True, allow_unused=True)

        with torch.no_grad():
            for wk, gw in zip(w_list, grads_w):
                if gw is None:
                    continue
                wk.data -= w_lr * gw
                wk.data.clamp_(min=self.w_floor)

            self._normalize(weight_pdict, active, budget)

    def forward(self, outputs, labels, model=None, shared_params=None, return_details=True):
        if shared_params is None:
            if model is None:
                raise ValueError("Pass either `model` or `shared_params`.")
            shared_params = self._shared_params_from_model(model)

        comps = self._collect_components(outputs, labels)

        if not comps:
            device = None
            for v in outputs.values():
                if isinstance(v, torch.Tensor):
                    device = v.device; break
            if device is None:
                device = torch.device("cpu")
            zero = torch.tensor(0.0, requires_grad=True, device=device)
            return (zero, {}) if return_details else zero

        self._gradnorm_update_wallet(comps, self.SUP_KEYS, self.init_sup, self.weight_sup,
                                     self.alpha_sup, self.w_lr_sup, self.budget_sup, shared_params)
        self._gradnorm_update_wallet(comps, self.SSL_KEYS, self.init_ssl, self.weight_ssl,
                                     self.alpha_ssl, self.w_lr_ssl, self.budget_ssl, shared_params)

        total = 0.0
        for k in self.SUP_KEYS:
            if k in comps:
                total = total + self.weight_sup[k].detach() * comps[k]
        for k in self.SSL_KEYS:
            if k in comps:
                total = total + self.weight_ssl[k].detach() * comps[k]

        if not return_details:
            return total

        details = {
            "components": {k: self._safe_detach(v).item() for k, v in comps.items()},
            "weights_sup": {k: self._safe_detach(self.weight_sup[k]).item() for k in self.SUP_KEYS},
            "weights_ssl": {k: self._safe_detach(self.weight_ssl[k]).item() for k in self.SSL_KEYS},
            "budget_sup": self.budget_sup,
            "budget_ssl": self.budget_ssl,
        }
        return total, details
