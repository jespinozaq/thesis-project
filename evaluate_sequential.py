import json
import torch
import numpy as np
from model import BLAIRRecommender
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to directory containing BLAIR checkpoint (config.json + model.safetensors)')
    parser.add_argument('--model_name', default='roberta-base', help='Base model name for tokenizer (only used if needed)')
    parser.add_argument('--test_seq', required=True, help='JSON file with test sequences')
    parser.add_argument('--meta_file', required=True, help='JSONL with item metadata (parent_asin, etc.)')
    parser.add_argument('--item_pop', required=True, help='JSON with raw item popularity counts')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_meta(meta_path):
    """Return dict: asin -> metadata string."""
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item:
                parts.extend(item['features'])
            if 'description' in item:
                parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n', ' ').strip()
    return meta

def encode_all_items(model, tokenizer, item_asins, meta_dict, device, batch_size):
    """Encode all items and return numpy array [N, D] + list of asins in the same order."""
    model.eval()
    asins = list(item_asins)
    # Simple dataset for item encoding
    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len=64):
            self.asins = asins
            self.meta_dict = meta_dict
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]
            meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'asin': asin}

    dataset = ItemDataset(asins, meta_dict, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb = []
    all_asins = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding items"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            emb = model.encode_item(ids, mask)
            all_emb.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    emb_matrix = np.vstack(all_emb).astype('float32')
    # Normalize for cosine similarity (items already normalized in encode_item, but ensure)
    faiss.normalize_L2(emb_matrix)
    return emb_matrix, all_asins

def main():
    args = parse_args()

    # Load the transformer backbone from the local BLAIR checkpoint
    backbone = AutoModel.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Build BLAIRRecommender using the same backbone config; since I'm going to overwrite its encoders
    model = BLAIRRecommender(args.model_name)
    model.context_encoder.transformer.load_state_dict(backbone.state_dict())
    model.item_encoder.transformer.load_state_dict(backbone.state_dict())
    model.to(args.device)
    model.eval()

    # Load metadata and popularity
    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)
    # Compute log popularity (add 1 smoothing to avoid log(0))
    item_logpop = {asin: np.log(raw_pop.get(asin, 0) + 1) for asin in meta_dict.keys()}

    # Load test sequences
    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    # Build item index
    all_asins = list(meta_dict.keys())
    item_emb, index_asins = encode_all_items(model, tokenizer, all_asins, meta_dict,
                                             args.device, args.batch_size)
    # Build FAISS index
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product = cosine similarity because embeddings are L2-normalized
    index.add(item_emb)

    # Evaluation
    hits = defaultdict(int)   # k -> count
    ndcg = defaultdict(float)
    pop_biases = []
    diversities = []
    recommended_items = []    # for coverage

    for uid, data in tqdm(test_seqs.items(), desc="Evaluating users"):
        history = data['history']
        target_asin = data['target_asin']

        # Get embeddings of history items
        hist_embs = []
        for asin in history:
            if asin in meta_dict and asin in index_asins:
                idx = index_asins.index(asin)
                emb = item_emb[idx]   # 1D vector
                hist_embs.append(emb)
        if not hist_embs:
            # Cold-start: use average of all items (or zero)
            user_emb = np.mean(item_emb, axis=0)
        else:
            user_emb = np.mean(hist_embs, axis=0)

        # Normalize user embedding
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm

        # Search top-K
        D, I = index.search(np.expand_dims(user_emb.astype('float32'), 0), args.topk)
        retrieved_asins = [index_asins[i] for i in I[0]]

        # Metrics
        if target_asin in retrieved_asins:
            rank = retrieved_asins.index(target_asin) + 1
            for k in range(1, args.topk+1):
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)

        # Popularity bias: log popularity difference
        retrieved_logpops = [item_logpop.get(a, 0) for a in retrieved_asins]
        rec_avg_logpop = np.mean(retrieved_logpops)
        hist_logpops = [item_logpop.get(a, 0) for a in history]
        if hist_logpops:
            user_avg_logpop = np.mean(hist_logpops)
        else:
            user_avg_logpop = 0.0
        pop_biases.append(rec_avg_logpop - user_avg_logpop)

        # Intra-list diversity: 1 - avg pairwise cosine similarity
        retrieved_embs = np.array([item_emb[index_asins.index(a)] for a in retrieved_asins])
        # Compute pairwise similarities
        sim_matrix = np.dot(retrieved_embs, retrieved_embs.T)  # all vectors normalized
        np.fill_diagonal(sim_matrix, 0)
        if len(retrieved_embs) > 1:
            # Average over off-diagonal elements
            offdiag = sim_matrix[~np.eye(len(sim_matrix), dtype=bool)]
            avg_sim = offdiag.mean()
        else:
            avg_sim = 0.0
        diversities.append(1.0 - avg_sim)

        recommended_items.extend(retrieved_asins)

    # Print results
    total = len(test_seqs)
    print(f"\nEvaluation on {total} test users (top-{args.topk}):")
    for k in [1, 5, 10]:
        if k <= args.topk:
            hr = hits[k] / total
            nd = ndcg[k] / total
            print(f"HR@{k}: {hr:.4f}  NDCG@{k}: {nd:.4f}")

    avg_pop_bias = np.mean(pop_biases)
    print(f"Average Log Popularity Difference: {avg_pop_bias:.4f}")

    avg_diversity = np.mean(diversities)
    print(f"Average Intra-list Diversity (1 - cosine similarity): {avg_diversity:.4f}")

    # Coverage: fraction of all items that appear in any recommendation list
    unique_recs = set(recommended_items)
    coverage = len(unique_recs) / len(all_asins) if all_asins else 0
    print(f"Catalogue Coverage: {coverage:.4f} ({len(unique_recs)}/{len(all_asins)} items)")

if __name__ == '__main__':
    main()