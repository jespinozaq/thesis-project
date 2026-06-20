import json, torch, argparse, numpy as np, faiss
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from model import BLAIRRecommender
import os

def load_meta(meta_path):
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title','')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n',' ').strip()
    return meta

def build_item_index(model, meta_dict, device, batch_size, max_seq_length, model_name):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Fix for Llama tokenizer
    if "llama" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len):
            self.asins = asins; self.meta_dict = meta_dict; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]; meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'asin': asin}
    loader = torch.utils.data.DataLoader(ItemDataset(list(meta_dict.keys()), meta_dict, tokenizer, max_len=max_seq_length),
                                         batch_size=batch_size, shuffle=False)
    all_embs, all_asins = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building item index"):
            emb = model.encode_item(batch['input_ids'].to(device),
                                    batch['attention_mask'].to(device))
            all_embs.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    all_embs = np.vstack(all_embs).astype('float32')
    index = faiss.IndexFlatIP(all_embs.shape[1])
    index.add(all_embs)
    return index, all_asins, all_embs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--test_seq', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop', required=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--output_tag', default='model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)
    item_freq = {asin: raw_pop.get(asin, 0) for asin in meta_dict}

    freqs = sorted(item_freq.values())
    n = len(freqs)
    low_thr = freqs[int(n * 0.33)] if n > 0 else 0
    high_thr = freqs[int(n * 0.66)] if n > 0 else 0

    def quantile(asin):
        f = item_freq.get(asin, 0)
        if f >= high_thr: return 'head'
        elif f >= low_thr: return 'mid'
        else: return 'tail'

    # Load model – handle LLM checkpoints
    if 'llama' in args.model_name.lower():
        from model_llm import LLMRecommender
        model = LLMRecommender(args.model_name)
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state, strict=False)
    else:
        model = BLAIRRecommender(args.model_name)
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    asins_list = list(meta_dict.keys())
    item_index, index_asins, item_embs = build_item_index(
        model, meta_dict, args.device,
        batch_size=256, max_seq_length=64, model_name=args.model_name)

    hits_by_pop = defaultdict(int)
    total_by_pop = defaultdict(int)
    all_recs = {}

    for uid, data in tqdm(test_seqs.items(), desc="Evaluating users"):
        history = data['history']
        target = data['target_asin']

        hist_embs = []
        for a in history:
            if a in meta_dict and a in index_asins:
                idx = index_asins.index(a)
                hist_embs.append(item_embs[idx])
        if hist_embs:
            user_emb = np.mean(hist_embs, axis=0)
        else:
            user_emb = np.mean(item_embs, axis=0)
        norm = np.linalg.norm(user_emb)
        if norm > 0:
            user_emb /= norm

        D, I = item_index.search(np.expand_dims(user_emb.astype('float32'), 0), args.topk)
        retrieved = [index_asins[i] for i in I[0]]

        pop_cat = quantile(target)
        total_by_pop[pop_cat] += 1
        if target in retrieved:
            hits_by_pop[pop_cat] += 1

        all_recs[uid] = {
            'history': history,
            'recommended': retrieved,
            'target': target
        }

    print(f"\nStratified Recall@{args.topk} ({args.output_tag}):")
    for cat in ['head', 'mid', 'tail']:
        hr = hits_by_pop[cat] / total_by_pop[cat] if total_by_pop[cat] > 0 else 0.0
        print(f"  {cat}: {hr:.4f} ({hits_by_pop[cat]}/{total_by_pop[cat]})")

    out_json = f"recommendations_{args.output_tag}.json"
    with open(out_json, 'w') as f:
        json.dump(all_recs, f)
    print(f"Saved per-user recommendations to {out_json}")

if __name__ == '__main__':
    main()
