import torch
import torch.nn.functional as F

def scaled_cross_entropy_loss(context_emb, item_emb, item_freqs, temperature=0.05, beta=0.2):
    """
    Scaled Cross-Entropy (SCE) with frequency-based scaling of negative logits.
    context_emb: [B, D]
    item_emb: [B, D]
    item_freqs: [B] raw frequencies of the positive items in the batch
    """
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature  # [B, B]

    max_freq = item_freqs.max()
    weights = (item_freqs / max_freq).pow(beta).unsqueeze(0)  # [1, B]
    mask = torch.eye(B, device=sim.device).bool()
    neg_weights = weights.expand(B, B).clone()
    neg_weights[mask] = 1.0  # don't scale positive
    sim_scaled = sim - torch.log(neg_weights + 1e-8)

    labels = torch.arange(B, device=sim.device)
    loss = F.cross_entropy(sim_scaled, labels)
    return loss

def popularity_penalty(topk_item_logpops, user_hist_logpop):
    """
    Penalize recommending items with higher log popularity than user's history.

    topk_item_logpops: [B, K] log popularity of top-K items per user
    user_hist_logpop:  [B] average log popularity of user's historical items

    This is a standard relu-based calibration penalty used in the literature
    (cf. SPREE / Popularity Quantile Calibration, FAccT 2026).
    During training, user_hist_logpop is approximated by the log-popularity
    of the positive item in each batch—a one-pass approximation that is
    consistent with the Calibrated Popularity framework.
    """
    rec_avg = topk_item_logpops.mean(dim=1)  # [B]
    penalty = F.relu(rec_avg - user_hist_logpop)
    return penalty.mean()

def diversity_regularizer(topk_item_emb):
    """
    Encourage intra-list diversity: negative average pairwise cosine similarity.
    topk_item_emb: [B, K, D]
    """
    B, K, D = topk_item_emb.shape
    sim_matrix = torch.bmm(topk_item_emb, topk_item_emb.transpose(1, 2))  # [B, K, K]
    mask = torch.eye(K, device=topk_item_emb.device).bool().unsqueeze(0)
    off_diag = sim_matrix.masked_select(~mask).view(B, K, K-1)
    avg_sim = off_diag.mean(dim=(1,2))  # [B]
    return -avg_sim.mean()  # minimize -> maximize diversity
