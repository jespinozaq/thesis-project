#!/usr/bin/env python3
"""LLM candidate‑selection recommender with parallel API calls."""

import json, random, argparse, re, time, sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI as OpenAIClient
except ImportError:
    OpenAIClient = None

import torch
from transformers import AutoModel, AutoTokenizer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_seqs', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop', required=True)
    parser.add_argument('--backend', required=True, choices=['openai', 'anthropic', 'ollama'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt_mode', default='base', choices=['base', 'mitigate', 'minimize', 'fair'])
    parser.add_argument('--api_key', default=None)
    parser.add_argument('--base_url', default=None)
    parser.add_argument('--num_candidates', type=int, default=100)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--blair_model_path', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=1, help='Parallel workers for LLM calls')
    parser.add_argument('--request_delay', type=float, default=0.5, help='Delay between calls (seconds)')
    return parser.parse_args()

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

def build_history_str(history_asins, meta_dict):
    items = [f"- {meta_dict.get(a, '')[:120]}" for a in history_asins[-10:] if a in meta_dict]
    return "\n".join(items) if items else "None"

def prepare_candidates(target_asin, history_asins, all_asins, meta_dict, num_candidates):
    history_set = set(history_asins)
    neg_pool = [a for a in all_asins if a != target_asin and a not in history_set]
    if len(neg_pool) < num_candidates - 1:
        neg_samples = neg_pool
    else:
        neg_samples = random.sample(neg_pool, num_candidates - 1)
    cands = [target_asin] + neg_samples
    random.shuffle(cands)
    return cands

def prompt_candidates(candidates, meta_dict):
    lines = [f"{i+1}. {meta_dict.get(a, '')[:150]}" for i, a in enumerate(candidates)]
    return "\n".join(lines)

def build_ranking_prompt(history_str, candidates_str, prompt_mode, topk):
    base = (
        "You are a helpful shopping assistant. A user has previously viewed/purchased these items:\n"
        f"{history_str}\n\n"
        "Here is a list of candidate items:\n"
        f"{candidates_str}\n\n"
        f"Please list the {topk} items that would be the best next recommendations for this user, "
        "in order of preference, as item numbers separated by commas (e.g., 3,17,5,9,...)."
    )
    if prompt_mode == 'base':
        return base
    elif prompt_mode == 'fair':
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
    else:
        return base

def query_openai(prompt, model, api_key, base_url=None):
    kwargs = {"api_key": api_key}
    if base_url: kwargs["base_url"] = base_url
    client = OpenAIClient(**kwargs)
    resp = client.chat.completions.create(
        model=model, messages=[{"role":"user","content":prompt}],
        temperature=0.0, max_tokens=50
    )
    return resp.choices[0].message.content.strip()

def process_single_user(task):
    uid, candidates, prompt, backend, model, api_key, base_url, topk, request_delay = task
    time.sleep(request_delay)
    out = None
    for attempt in range(3):
        try:
            if backend == 'openai':
                out = query_openai(prompt, model, api_key, base_url)
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

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)
    item_logpop = {a: np.log(raw_pop.get(a, 0) + 1) for a in meta_dict}
    all_asins = list(meta_dict.keys())

    tokenizer = AutoTokenizer.from_pretrained(args.blair_model_path)
    blair_model = AutoModel.from_pretrained(args.blair_model_path).to(args.device)
    blair_model.eval()
    print("Precomputing item embeddings...")
    all_emb = []
    bs = 256
    for i in tqdm(range(0, len(all_asins), bs), desc="Encoding items"):
        batch_asins = all_asins[i:i+bs]
        texts = [meta_dict[a] for a in batch_asins]
        enc = tokenizer(texts, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
        with torch.no_grad():
            out = blair_model(input_ids=enc['input_ids'].to(args.device),
                              attention_mask=enc['attention_mask'].to(args.device))
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_emb.append(emb.cpu().numpy())
    all_emb = np.concatenate(all_emb, axis=0).astype('float32')
    asin_to_idx = {a: idx for idx, a in enumerate(all_asins)}

    with open(args.test_seqs, 'r') as f:
        test_seqs = json.load(f)

    tasks = []
    for uid, data in test_seqs.items():
        history = data['history']
        target = data['target_asin']
        candidates = prepare_candidates(target, history, all_asins, meta_dict, args.num_candidates)
        cand_str = prompt_candidates(candidates, meta_dict)
        hist_str = build_history_str(history, meta_dict)
        prompt = build_ranking_prompt(hist_str, cand_str, args.prompt_mode, args.topk)
        tasks.append((uid, candidates, prompt, args.backend, args.model, args.api_key, args.base_url, args.topk, args.request_delay))

    user_results = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_user, t): t[0] for t in tasks}
        with tqdm(total=len(futures), desc="LLM calls") as pbar:
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
        rec_logpops = [item_logpop.get(a, 0) for a in retrieved]
        rec_avg = np.mean(rec_logpops)
        hist_logpops = [item_logpop.get(a, 0) for a in data['history']]
        user_avg = np.mean(hist_logpops) if hist_logpops else 0.0
        pop_diffs.append(rec_avg - user_avg)
        retrieved_embs = []
        for a in retrieved:
            if a in asin_to_idx:
                retrieved_embs.append(all_emb[asin_to_idx[a]])
        if len(retrieved_embs) > 1:
            embs = np.array(retrieved_embs)
            sim = np.dot(embs, embs.T)
            np.fill_diagonal(sim, 0)
            offdiag = sim[~np.eye(len(sim), dtype=bool)]
            diversities.append(1.0 - offdiag.mean())
        else:
            diversities.append(0.0)
        recommended_items_global.extend(retrieved)

    print(f"\nLLM Recommender: backend={args.backend}, model={args.model}, prompt={args.prompt_mode}")
    print(f"Total test users: {total}")
    for k in range(1, args.topk+1):
        hr_k = hits[k] / total if total else 0
        nd_k = ndcg[k] / total if total else 0
        print(f"HR@{k}: {hr_k:.4f}  NDCG@{k}: {nd_k:.4f}")
    avg_pop = np.mean(pop_diffs) if pop_diffs else 0
    print(f"Average Log Popularity Difference: {avg_pop:.4f}")
    avg_div = np.mean(diversities) if diversities else 0
    print(f"Average Intra-list Diversity (1 - cosine similarity): {avg_div:.4f}")
    unique_recs = set(recommended_items_global)
    coverage = len(unique_recs) / len(all_asins) if all_asins else 0
    print(f"Catalogue Coverage: {coverage:.4f} ({len(unique_recs)}/{len(all_asins)} items)")

if __name__ == '__main__':
    main()
