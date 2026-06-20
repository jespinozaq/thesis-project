import json, torch, numpy as np, faiss, argparse, os
from collections import Counter
from tqdm import tqdm
from model import BLAIRRecommender
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--test_seq', required=True)
    parser.add_argument('--meta_file', required=True)
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
            emb = model.encode_item(batch['input_ids'].to(device),
                                    batch['attention_mask'].to(device))
            all_emb.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    emb_matrix = np.vstack(all_emb).astype('float32')
    faiss.normalize_L2(emb_matrix)
    return emb_matrix, all_asins

def gini(array):
    if len(array) == 0 or array.sum() == 0:
        return 0.0
    sorted_array = np.sort(array)
    n = len(sorted_array)
    index = np.arange(1, n+1)
    return (2 * np.sum(index * sorted_array) - (n+1)*np.sum(sorted_array)) / (n * np.sum(sorted_array))

def herfindahl(array):
    total = array.sum()
    if total == 0:
        return 0.0
    shares = array / total
    return np.sum(shares ** 2)

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "llama" in args.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # Model loading – directory, fine‑tuned, or llama
    if os.path.isdir(args.model_path):
        model = BLAIRRecommender(args.model_path)
    elif "llama" in args.model_name.lower():
        from model_llm import LLMRecommender
        model = LLMRecommender(args.model_name)
        state = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state, strict=False)
    else:
        model = BLAIRRecommender(args.model_name)
        state = torch.load(args.model_path, map_location=args.device)
        model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    meta_dict = load_meta(args.meta_file)
    all_asins = list(meta_dict.keys())
    item_emb, index_asins = encode_all_items(model, tokenizer, all_asins, meta_dict,
                                             args.device, args.batch_size)
    dim = item_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(item_emb)

    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    rec_counter = Counter()

    for uid, data in tqdm(test_seqs.items(), desc="Retrieving for provider fairness"):
        history = data['history']
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
        for asin in retrieved:
            rec_counter[asin] += 1

    counts = np.array(list(rec_counter.values()), dtype=np.float64)
    total_recs = len(test_seqs) * args.topk
    num_unique = len(rec_counter)
    coverage = num_unique / len(all_asins) if all_asins else 0

    print(f"\nProvider‑Side Exposure Fairness (top-{args.topk})")
    print(f"Total recommendations issued: {total_recs}")
    print(f"Unique items recommended: {num_unique} (Coverage: {coverage:.4f})")
    print(f"Gini coefficient of item exposure: {gini(counts):.4f}")
    print(f"Herfindahl-Hirschman Index (HHI): {herfindahl(counts):.4f}")

if __name__ == '__main__':
    main()
