#!/usr/bin/env python3
"""Re‑rank top‑100 candidates with parallel LLM calls."""

import json, torch, numpy as np, faiss, argparse, re, time
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import BLAIRRecommender
from transformers import AutoTokenizer

def query_openai(prompt, model, api_key, base_url=None):
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    if base_url: kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=model, messages=[{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=50
    )
    return resp.choices[0].message.content.strip()

def load_meta(meta_path):
    meta = {}
    with open(meta_path,'r') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title','')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n',' ').strip()
    return meta

def build_history_str(history_asins, meta_dict):
    lines = [f"- {meta_dict.get(a, '')[:120]}" for a in history_asins[-10:]]
    return "\n".join(lines) if lines else "None"

def build_candidate_str(candidates, meta_dict):
    lines = [f"{i+1}. {meta_dict.get(a, '')[:150]}" for i, a in enumerate(candidates)]
    return "\n".join(lines)

def build_promptB(history_str, candidates_str, topk):
    return (
        "We want to make a fair recommendation for this user based on their history.\n"
        f"User history:\n{history_str}\n\n"
        "Candidates:\n"
        f"{candidates_str}\n\n"
        f"Select the {topk} best items that are relevant to the user's history. "
        "When choosing, consider including a mix of both popular and less well‑known items, "
        "and try to offer some variety across different types of items. "
        "Do not avoid popular items entirely, but also do not choose them exclusively.\n\n"
        "Output the numbers of the selected items in order of preference, separated by commas (e.g., 3,17,5)."
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model_name', default='roberta-base')
    parser.add_argument('--test_seq', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--candidate_pool', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--request_delay', type=float, default=0.5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = BLAIRRecommender(args.model_name)
    state = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop,'r') as f:
        raw_pop = json.load(f)
    item_logpop = {a: np.log(raw_pop.get(a,0)+1) for a in meta_dict}

    with open(args.test_seq,'r') as f:
        test_seqs = json.load(f)

    all_asins = list(meta_dict.keys())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    class ItemDs(torch.utils.data.Dataset):
        def __init__(self, asins, meta, tok, max_len=64):
            self.asins = asins; self.meta = meta; self.tok = tok; self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            a = self.asins[idx]; m = self.meta[a]
            enc = self.tok(m, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            return {'input_ids':enc['input_ids'].squeeze(0), 'attention_mask':enc['attention_mask'].squeeze(0), 'asin':a}
    dl = torch.utils.data.DataLoader(ItemDs(all_asins, meta_dict, tokenizer), batch_size=256, shuffle=False)
    emb_list, asin_list = [], []
    with torch.no_grad():
        for b in tqdm(dl, desc="Encoding items"):
            e = model.encode_item(b['input_ids'].to(args.device), b['attention_mask'].to(args.device))
            emb_list.append(e.cpu().numpy())
            asin_list.extend(b['asin'])
    emb_mat = np.vstack(emb_list).astype('float32')
    faiss.normalize_L2(emb_mat)
    index = faiss.IndexFlatIP(emb_mat.shape[1])
    index.add(emb_mat)
    asin_to_idx = {a:i for i,a in enumerate(asin_list)}

    tasks = []
    for uid, data in test_seqs.items():
        history = data['history']
        target = data['target_asin']
        hist_embs = []
        for a in history:
            if a in asin_to_idx:
                hist_embs.append(emb_mat[asin_to_idx[a]])
        if hist_embs:
            user_emb = np.mean(hist_embs, axis=0)
        else:
            user_emb = np.mean(emb_mat, axis=0)
        user_emb /= np.linalg.norm(user_emb)
        D, I = index.search(np.expand_dims(user_emb.astype('float32'), 0), args.candidate_pool)
        candidates = [asin_list[i] for i in I[0]]
        hist_str = build_history_str(history, meta_dict)
        cand_str = build_candidate_str(candidates, meta_dict)
        prompt = build_promptB(hist_str, cand_str, args.topk)
        tasks.append((uid, candidates, prompt, args.backend, args.llm_model, args.api_key, args.base_url, args.topk, args.request_delay))

    def process(task):
        uid, candidates, prompt, backend, model_llm, key, url, topk, delay = task
        time.sleep(delay)
        out = None
        for attempt in range(3):
            try:
                if backend == 'openai':
                    out = query_openai(prompt, model_llm, key, url)
                break
            except Exception as e:
                print(f"Retry {attempt} for {uid}: {e}")
                time.sleep(2)
        if out is None:
            return uid, candidates[:topk]
        numbers = re.findall(r'\d+', out)
        idxs = [int(n)-1 for n in numbers if 1 <= int(n) <= len(candidates)]
        idxs = list(dict.fromkeys(idxs))[:topk]
        while len(idxs) < topk:
            for i in range(len(candidates)):
                if i not in idxs:
                    idxs.append(i)
                    if len(idxs)>=topk: break
        return uid, [candidates[i] for i in idxs[:topk]]

    user_results = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process, t): t[0] for t in tasks}
        with tqdm(total=len(futures), desc="LLM re‑ranking") as pbar:
            for future in as_completed(futures):
                uid, recs = future.result()
                user_results[uid] = recs
                pbar.update(1)

    total = len(test_seqs)
    hits = defaultdict(int)
    ndcg = defaultdict(float)
    pop_diffs = []
    diversities = []
    recommended_items_global = []

    for uid, data in test_seqs.items():
        target = data['target_asin']
        retrieved = user_results[uid]
        if target in retrieved:
            rank = retrieved.index(target) + 1
            for k in range(1, args.topk+1):
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)
        rec_logpops = [item_logpop[a] for a in retrieved]
        hist_logpops = [item_logpop[a] for a in data['history'] if a in item_logpop]
        pop_diffs.append(np.mean(rec_logpops) - (np.mean(hist_logpops) if hist_logpops else np.mean(rec_logpops)))
        retrieved_embs = [emb_mat[asin_to_idx[a]] for a in retrieved if a in asin_to_idx]
        if len(retrieved_embs) > 1:
            embs = np.array(retrieved_embs)
            sim = np.dot(embs, embs.T)
            np.fill_diagonal(sim, 0)
            offdiag = sim[~np.eye(len(sim), dtype=bool)]
            diversities.append(1.0 - offdiag.mean())
        else:
            diversities.append(0.0)
        recommended_items_global.extend(retrieved)

    print(f"\nCombined re‑ranking results (top-{args.topk}):")
    for k in range(1, args.topk+1):
        hr_k = hits[k]/total if total else 0
        nd_k = ndcg[k]/total if total else 0
        print(f"HR@{k}: {hr_k:.4f}  NDCG@{k}: {nd_k:.4f}")
    print(f"Average Log Popularity Difference: {np.mean(pop_diffs):.4f}")
    print(f"Average Intra‑list Diversity: {np.mean(diversities):.4f}")
    coverage = len(set(recommended_items_global))/len(all_asins)
    print(f"Catalogue Coverage: {coverage:.4f}")

if __name__ == '__main__':
    main()
