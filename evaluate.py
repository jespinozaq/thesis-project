import torch
from torch.utils.data import DataLoader
from data_utils import AmazonReviewDataset, collate_fn
from model import BLAIRRecommender
from transformers import AutoTokenizer
import faiss
import numpy as np

def evaluate_sequential(model, test_loader, item_index, item_asins, k=10):
    model.eval()
    hits = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            c_emb = model.encode_context(c_ids, c_mask)
            # Retrieve top-k
            D, I = item_index.search(c_emb.cpu().numpy(), k)
            for i, true_asin in enumerate(batch['item_asin']):
                retrieved_asins = [item_asins[idx] for idx in I[i]]
                if true_asin in retrieved_asins:
                    hits += 1
                total += 1
    recall = hits / total
    print(f"Recall@{k}: {recall:.4f}")
    return recall