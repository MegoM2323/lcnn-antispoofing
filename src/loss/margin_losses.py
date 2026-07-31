"""
Margin-based losses for voice anti-spoofing (arXiv:2103.11326, Sec. 2.2).

All of them share the generalized softmax formulation (Eq. 2):

    P_{j,k} = exp(a * [cos(m1 * theta_{j,k} + m2) - m3])
              / (exp(a * [cos(m1 * theta_{j,k} + m2) - m3])
                 + sum_{i != k} exp(a * cos theta_{j,i}))

where theta_{j,i} is the angle between the embedding of sample j and the class
vector i, k is the target class and a is the scale.

IMPORTANT (shared by every loss in this module): the returned dict contains a
"logits" key with the cosine scores of the batch. The trainer performs
batch.update(outputs) (model) and only then batch.update(all_losses)
(loss), so this key *overwrites* the logits produced by the model. This is
intentional: with margin losses the class vectors live inside the loss module,
so the cosine scores computed here are the only calibrated scores available,
and the metrics (EER, accuracy) must see them instead of the raw model head
output. Keep this in mind when a margin loss is combined with a model that has
its own classification head.
"""

import torch
import torch.nn.functional as F
from torch import nn


def cosine_scores(embedding: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity between L2-normalized embeddings and L2-normalized class
    vectors.

    Args:
        embedding (Tensor): batch of embeddings of shape (B, D).
        weight (Tensor): class vectors of shape (D, n_classes).
    Returns:
        cosine (Tensor): cosine similarities of shape (B, n_classes), in [-1, 1].
    """
    if embedding.ndim != 2:
        raise ValueError(
            f"Expected embedding of shape (B, D), got {tuple(embedding.shape)}"
        )
    return F.normalize(embedding, dim=-1) @ F.normalize(weight, dim=0)


class AMSoftmaxLoss(nn.Module):
    """
    Additive Margin Softmax: m1 = 1, m2 = 0, m3 = margin.

    The target logit is penalized by a fixed additive margin in the cosine
    domain, which forces an angular gap between bonafide and spoof clusters.
    Defaults (margin = 0.9, scale = 20) are the ones reported in
    arXiv:2103.11326 for the ASVspoof2019 LA setup.
    """

    def __init__(
        self,
        embed_dim: int,
        n_classes: int = 2,
        margin: float = 0.9,
        scale: float = 20.0,
    ):
        """
        Args:
            embed_dim (int): dimensionality D of the model embedding.
            n_classes (int): number of classes (2 for spoof/bonafide).
            margin (float): additive cosine margin m3.
            scale (float): softmax scale alpha.
        """
        super().__init__()
        if embed_dim <= 0 or n_classes < 2:
            raise ValueError(
                f"Invalid shapes: embed_dim={embed_dim}, n_classes={n_classes}"
            )
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(embed_dim, n_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embedding (Tensor): model embedding of shape (B, D).
            labels (Tensor): ground-truth labels of shape (B,).
        Returns:
            losses (dict): 'loss' and 'logits' (scaled cosine scores, computed
                without the margin; see the module docstring).
        """
        cosine = cosine_scores(embedding, self.weight)
        target_mask = F.one_hot(labels, num_classes=cosine.shape[-1]).to(cosine.dtype)
        margin_logits = self.scale * (cosine - self.margin * target_mask)
        loss = F.cross_entropy(margin_logits, labels)
        return {"loss": loss, "logits": self.scale * cosine.detach()}


class OCSoftmaxLoss(nn.Module):
    """
    One-Class Softmax (Zhang et al., 2021).

    Instead of one vector per class, a single center w0 is learned for the
    bonafide class, and two different margins are applied: bonafide embeddings
    are pushed to have cos >= m_real, spoofed ones to have cos <= m_fake.
    Since spoofing attacks are open-set, this compacts the bonafide cluster
    without forcing all (unseen) attacks into a single cluster:

        L = mean_j softplus(alpha * s_j * (m_{y_j} - cos theta_j)),
        s_j = +1 for bonafide (label 1), -1 for spoof (label 0)

    Defaults follow arXiv:2103.11326: alpha = 20, m_real = 0.9, m_fake = 0.2.
    """

    def __init__(
        self,
        embed_dim: int,
        margin_real: float = 0.9,
        margin_fake: float = 0.2,
        scale: float = 20.0,
    ):
        """
        Args:
            embed_dim (int): dimensionality D of the model embedding.
            margin_real (float): margin m3,1 for the bonafide (target) class.
            margin_fake (float): margin m3,2 for the spoofed (non-target) class.
            scale (float): scale alpha.
        """
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"Invalid embed_dim={embed_dim}")
        if margin_fake >= margin_real:
            raise ValueError(
                f"margin_fake ({margin_fake}) must be smaller than "
                f"margin_real ({margin_real})"
            )
        self.margin_real = margin_real
        self.margin_fake = margin_fake
        self.scale = scale
        # decision boundary sits between the two margins, so that argmax over
        # the exported logits is a sensible (if not tuned) hard decision
        self.decision_threshold = (margin_real + margin_fake) / 2
        self.center = nn.Parameter(torch.empty(embed_dim, 1))
        nn.init.xavier_normal_(self.center)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embedding (Tensor): model embedding of shape (B, D).
            labels (Tensor): ground-truth labels of shape (B,), 1 = bonafide.
        Returns:
            losses (dict): 'loss' and 'logits' of shape (B, 2), built as
                [-s, +s] with s = scale * (cos - decision_threshold), so that
                logits[:, 1] - logits[:, 0] is monotone in the cosine score
                (see the module docstring).
        """
        cosine = cosine_scores(embedding, self.center).squeeze(-1)

        is_bonafide = labels == 1
        margins = torch.where(
            is_bonafide,
            torch.full_like(cosine, self.margin_real),
            torch.full_like(cosine, self.margin_fake),
        )
        # +1 pushes cos above the margin (bonafide), -1 pushes it below (spoof)
        signs = torch.where(
            is_bonafide, torch.ones_like(cosine), -torch.ones_like(cosine)
        )
        loss = F.softplus(self.scale * signs * (margins - cosine)).mean()

        score = self.scale * (cosine - self.decision_threshold)
        logits = torch.stack((-score, score), dim=-1).detach()
        return {"loss": loss, "logits": logits}


class P2SGradLoss(nn.Module):
    """
    P2SGrad MSE loss (Wang et al., 2019): hyperparameter-free, no margin and no
    scale:

        L = (1 / |D|) * sum_j sum_k (cos theta_{j,k} - 1(y_j = k))^2

    The gradient of this MSE w.r.t. the cosine equals the P2SGrad gradient of
    the softmax cross-entropy, which removes the sensitivity to the scale and
    margin hyperparameters. It gave the best EER (1.92%) in arXiv:2103.11326.
    """

    def __init__(self, embed_dim: int, n_classes: int = 2):
        """
        Args:
            embed_dim (int): dimensionality D of the model embedding.
            n_classes (int): number of classes (2 for spoof/bonafide).
        """
        super().__init__()
        if embed_dim <= 0 or n_classes < 2:
            raise ValueError(
                f"Invalid shapes: embed_dim={embed_dim}, n_classes={n_classes}"
            )
        self.weight = nn.Parameter(torch.empty(embed_dim, n_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(
        self, embedding: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            embedding (Tensor): model embedding of shape (B, D).
            labels (Tensor): ground-truth labels of shape (B,).
        Returns:
            losses (dict): 'loss' and 'logits' (raw cosine scores, no scaling
                is needed since EER is invariant to monotone transforms; see
                the module docstring).
        """
        cosine = cosine_scores(embedding, self.weight)
        target = F.one_hot(labels, num_classes=cosine.shape[-1]).to(cosine.dtype)
        # sum over classes, mean over the batch (1/|D| in the paper)
        loss = ((cosine - target) ** 2).sum(dim=-1).mean()
        return {"loss": loss, "logits": cosine.detach()}
