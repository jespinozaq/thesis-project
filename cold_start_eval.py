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
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--test_seq', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop', required=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_meta(meta_path):
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n', ' ').strip()
    return meta

def encode_all_items(model, tokenizer, item_asins, meta_dict, device, batch_size):
    model.eval()
    asins = list(item_asins)
    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len=64):
            self.asins = asins; self.meta_dict = meta_dict; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]; meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'asin': asin}
    dataset = ItemDataset(asins, meta_dict, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb, all_asins = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding items"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            emb = model.encode_item(ids, mask)
            all_emb.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    emb_matrix = np.vstack(all_emb).astype('float32')
    faiss.normalize_L2(emb_matrix)
    return emb_matrix, all_asins

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BLAIRRecommender(args.model_name)
    state_dict = torch.load(args.model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)

    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    all_asins = list(meta_dict.keys())
    item_emb, index_asins = encode_all_items(model, tokenizer, all_asins, meta_dict,
                                             args.device, args.batch_size)
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_emb)

    warm_hits = defaultdict(int)
    cold_hits = defaultdict(int)
    warm_ndcg = defaultdict(float)
    cold_ndcg = defaultdict(float)
    total_warm = 0
    total_cold = 0

    for uid, data in tqdm(test_seqs.items(), desc="Evaluating users (cold-start split)"):
        history = data['history']
        target_asin = data['target_asin']
        count = raw_pop.get(target_asin, 0)
        is_warm = count > 0

        hist_embs = []
        for a in history:
            if a in meta_dict and a in index_asins:
                idx = index_asins.index(a)
                hist_embs.append(item_emb[idx])
        if hist_embs:
            user_emb = np.mean(hist_embs, axis=0)
        else:
            user_emb = np.mean(item_emb, axis=0)
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb = user_emb / norm

        D, I = index.search(np.expand_dims(user_emb.astype('float32'), 0), args.topk)
        retrieved = [index_asins[i] for i in I[0]]

        if target_asin in retrieved:
            rank = retrieved.index(target_asin) + 1
            for k in range(1, args.topk+1):
                if rank <= k:
                    if is_warm:
                        warm_hits[k] += 1
                        warm_ndcg[k] += 1.0 / np.log2(rank + 1)
                    else:
                        cold_hits[k] += 1
                        cold_ndcg[k] += 1.0 / np.log2(rank + 1)

        if is_warm:
            total_warm += 1
        else:
            total_cold += 1

    print(f"\nCold-Start Evaluation (top-{args.topk})")
    print(f"Warm items: {total_warm}, Cold items: {total_cold}")
    print("\nWarm Item Performance:")
    for k in [1, 5, 10]:
        if k <= args.topk:
            hr = warm_hits[k] / total_warm if total_warm > 0 else 0
            ndcg = warm_ndcg[k] / total_warm if total_warm > 0 else 0
            print(f"HR@{k}: {hr:.4f}  NDCG@{k}: {ndcg:.4f}")
    print("\nCold Item Performance:")
    for k in [1, 5, 10]:
        if k <= args.topk:
            hr = cold_hits[k] / total_cold if total_cold > 0 else 0
            ndcg = cold_ndcg[k] / total_cold if total_cold > 0 else 0
            print(f"HR@{k}: {hr:.4f}  NDCG@{k}: {ndcg:.4f}")

if __name__ == '__main__':
    main()
