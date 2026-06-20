import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

class BLAIREncoder(nn.Module):
    """Sentence encoder based on RoBERTa (or any AutoModel)."""
    def __init__(self, model_name='roberta-base', pooling='cls'):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling = pooling  # 'cls' or 'mean'

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == 'cls':
            # Use [CLS] token
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            # Mean pooling over tokens (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_emb = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_emb / sum_mask
        return F.normalize(embeddings, p=2, dim=-1)  # normalize to unit sphere

class BLAIRRecommender(nn.Module):
    """Two-tower model for context and item encoding."""
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.context_encoder = BLAIREncoder(model_name)
        self.item_encoder = BLAIREncoder(model_name)

    def encode_context(self, input_ids, attention_mask):
        return self.context_encoder(input_ids, attention_mask)

    def encode_item(self, input_ids, attention_mask):
        return self.item_encoder(input_ids, attention_mask)