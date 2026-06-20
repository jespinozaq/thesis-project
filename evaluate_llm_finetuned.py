import json, torch, numpy as np, faiss, argparse, re, warnings
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer
from model_llm import LLMRecommender

def load_meta(meta_path):
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n', ' ').strip()
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--test_seq', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop', required=True)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Model – load with strict=False to ignore extraneous quantisation metadata
    model = LLMRecommender(args.model_name)
    state_dict = torch.load(args.model_path, map_location='cpu')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys (will be initialized randomly): {len(missing)}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected quantisation metadata keys.")
    model.to(args.device)
    model.eval()

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)
    item_logpop = {asin: np.log(raw_pop.get(asin, 0) + 1) for asin in meta_dict}

    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    all_asins = list(meta_dict.keys())
    # Encode all items
    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len=128):
            self.asins = asins; self.meta_dict = meta_dict; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]; meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'asin': asin}
    loader = torch.utils.data.DataLoader(ItemDataset(all_asins, meta_dict, tokenizer), batch_size=args.batch_size, shuffle=False)
    emb_list, asin_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding items"):
            emb = model.encode_item(batch['input_ids'].to(args.device), batch['attention_mask'].to(args.device))
            emb_list.append(emb.cpu().numpy())
            asin_list.extend(batch['asin'])
    emb_matrix = np.vstack(emb_list).astype('float32')
    faiss.normalize_L2(emb_matrix)

    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    hits = defaultdict(int)
    ndcg = defaultdict(float)
    pop_diffs, diversities = [], []
    recommended_items_global = []

    for uid, data in tqdm(test_seqs.items(), desc="Evaluating"):
        history = data['history']; target = data['target_asin']
        hist_embs = []
        for a in history:
            if a in asin_list:
                idx = asin_list.index(a)
                hist_embs.append(emb_matrix[idx])
        if hist_embs:
            user_emb = np.mean(hist_embs, axis=0)
        else:
            user_emb = np.mean(emb_matrix, axis=0)
        user_emb /= np.linalg.norm(user_emb)

        D, I = index.search(np.expand_dims(user_emb.astype('float32'), 0), args.topk)
        retrieved = [asin_list[i] for i in I[0]]

        if target in retrieved:
            rank = retrieved.index(target) + 1
            for k in range(1, args.topk+1):
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)

        rec_logpops = [item_logpop[a] for a in retrieved]
        hist_logpops = [item_logpop[a] for a in history if a in item_logpop]
        pop_diffs.append(np.mean(rec_logpops) - (np.mean(hist_logpops) if hist_logpops else np.mean(rec_logpops)))

        retrieved_embs = np.array([emb_matrix[asin_list.index(a)] for a in retrieved])
        sim = np.dot(retrieved_embs, retrieved_embs.T)
        np.fill_diagonal(sim, 0)
        if len(retrieved_embs) > 1:
            offdiag = sim[~np.eye(len(retrieved_embs), dtype=bool)]
            diversities.append(1.0 - offdiag.mean())
        else:
            diversities.append(0.0)
        recommended_items_global.extend(retrieved)

    total = len(test_seqs)
    for k in [1,5,10]:
        if k <= args.topk:
            print(f"HR@{k}: {hits[k]/total:.4f}  NDCG@{k}: {ndcg[k]/total:.4f}")
    print(f"Average Log Popularity Difference: {np.mean(pop_diffs):.4f}")
    print(f"Average Intra-list Diversity (1 - cosine similarity): {np.mean(diversities):.4f}")
    print(f"Catalogue Coverage: {len(set(recommended_items_global))/len(all_asins):.4f} ({len(set(recommended_items_global))}/{len(all_asins)} items)")

if __name__ == '__main__':
    main()
