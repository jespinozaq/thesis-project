import torch
import torch.nn.functional as F

def inverse_propensity_weighted_sce(context_emb, item_emb, item_freqs, temperature=0.05, beta=0.2):
    """
    SCE loss where positive pairs are weighted by 1 / freq (inverse propensity).
    context_emb, item_emb: [B, D]  (L2‑normalized)
    item_freqs: [B] raw frequencies of the positive items
    """
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature   # [B, B]

    # Inverse propensity weights
    eps = 1e-8
    w = 1.0 / (item_freqs.float() + eps)        # [B]
    w = w / w.sum() * B                         # normalise so sum = B

    # ----- standard SCE scaling of negatives -----
    max_freq = item_freqs.max()
    freq_scale = (item_freqs.float() / (max_freq + eps)).pow(beta).unsqueeze(0)  # [1, B]
    mask = torch.eye(B, device=sim.device).bool()
    neg_weights = freq_scale.expand(B, B).clone()
    neg_weights[mask] = 1.0                     # don't scale positive
    sim_scaled = sim - torch.log(neg_weights + eps)

    # Per‑sample cross‑entropy
    labels = torch.arange(B, device=sim.device)
    ce = F.cross_entropy(sim_scaled, labels, reduction='none')   # [B]
    loss = (ce * w).mean()
    return loss


def augmentation_consistency_loss(orig_emb, aug_emb, temperature=0.07):
    """
    InfoNCE loss to align original and augmented item embeddings.
    orig_emb, aug_emb: [B, D] (L2‑normalised)
    """
    B = orig_emb.size(0)
    sim = torch.matmul(orig_emb, aug_emb.T) / temperature   # [B, B]
    labels = torch.arange(B, device=sim.device)
    loss = F.cross_entropy(sim, labels)
    return loss
