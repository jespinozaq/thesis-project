import os
import json, torch, numpy as np, faiss, argparse, re, time
from tqdm import tqdm
from collections import defaultdict
from model import BLAIRRecommender
from transformers import AutoTokenizer

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

def build_history_str(history_asins, meta_dict):
    lines = [f"- {meta_dict.get(a, '')[:120]}" for a in history_asins[-10:]]
    return "\n".join(lines) if lines else "None"

def query_llm(prompt, backend, model, api_key, base_url=None):
    if backend == 'openai':
        from openai import OpenAI
        kwargs = {"api_key": api_key}
        if base_url: kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model, messages=[{"role":"user","content":prompt}],
            temperature=0.0, max_tokens=50
        )
        return resp.choices[0].message.content.strip()
    else:
        raise NotImplementedError

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
    parser.add_argument('--keep', type=int, default=8)
    parser.add_argument('--candidate_pool', type=int, default=30)
    parser.add_argument('--baseline_only', action='store_true',
                        help='Only compute retriever top-K from the candidate pool (no LLM)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load fine-tuned model
    #model = BLAIRRecommender(args.model_name)
    #state = torch.load(args.model_path, map_location='cpu')
    #model.load_state_dict(state)
    #model.to(args.device)
    #model.eval()
    if os.path.isdir(args.model_path):
        model = BLAIRRecommender(args.model_path)
        model.to(args.device)
    else:
        model = BLAIRRecommender(args.model_name)
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state)
        model.to(args.device)
    model.eval()

    meta_dict = load_meta(args.meta_file)
    with open(args.item_pop, 'r') as f:
        raw_pop = json.load(f)
    item_logpop = {a: np.log(raw_pop.get(a,0)+1) for a in meta_dict}

    with open(args.test_seq, 'r') as f:
        test_seqs = json.load(f)

    all_asins = list(meta_dict.keys())
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Build item index
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

    hits = defaultdict(int)
    ndcg = defaultdict(float)
    pop_diffs = []
    diversities = []
    recommended_items_global = []
    total = 0

    for uid, data in tqdm(test_seqs.items(), desc="Swap‑re‑ranking" if not args.baseline_only else "Baseline (pool-only)"):
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
        # ❌ No forced injection of target

        if args.baseline_only:
            final_list = candidates[:args.topk]
        else:
            core = candidates[:args.keep]
            rest = candidates[args.keep:]
            history_str = build_history_str(history, meta_dict)
            rest_str = "\n".join([f"{i+1}. {meta_dict.get(a, '')[:150]}" for i, a in enumerate(rest)])

            prompt = (
                f"We have already selected the following {args.keep} items for this user based on relevance:\n"
                + "\n".join([f"- {meta_dict.get(a, '')[:150]}" for a in core]) +
                f"\n\nFrom the additional candidates below, choose {args.topk - args.keep} items that would best complement the existing list. "
                "Aim to add variety and ensure the final list is well‑rounded, without simply favouring the most popular items. "
                "Output the numbers of the selected items (from the list below, 1‑based), separated by commas.\n\n"
                "Additional candidates:\n" + rest_str
            )

            out = None
            for attempt in range(3):
                try:
                    out = query_llm(prompt, args.backend, args.llm_model, args.api_key, args.base_url)
                    break
                except Exception as e:
                    print(f"Retry {attempt} for {uid}: {e}")
                    time.sleep(2)
            if out is None:
                swap = rest[:args.topk - args.keep]
            else:
                numbers = re.findall(r'\d+', out)
                chosen_idx = [int(n)-1 for n in numbers if 1 <= int(n) <= len(rest)]
                chosen_idx = list(dict.fromkeys(chosen_idx))[:args.topk - args.keep]
                swap = [rest[i] for i in chosen_idx]

            final_list = core + swap
            if len(final_list) < args.topk:
                for a in rest:
                    if a not in final_list:
                        final_list.append(a)
                        if len(final_list) == args.topk:
                            break

        total += 1
        if target in final_list:
            rank = final_list.index(target) + 1
            for k in range(1, args.topk+1):
                if rank <= k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 1)

        rec_logpops = [item_logpop[a] for a in final_list]
        hist_logpops = [item_logpop[a] for a in history if a in item_logpop]
        avg_hist = np.mean(hist_logpops) if hist_logpops else np.mean(rec_logpops)
        pop_diffs.append(np.mean(rec_logpops) - avg_hist)

        retrieved_embs = [emb_mat[asin_to_idx[a]] for a in final_list if a in asin_to_idx]
        if len(retrieved_embs) > 1:
            embs = np.array(retrieved_embs)
            sim = np.dot(embs, embs.T)
            np.fill_diagonal(sim, 0)
            offdiag = sim[~np.eye(len(sim), dtype=bool)]
            diversities.append(1.0 - offdiag.mean())
        else:
            diversities.append(0.0)
        recommended_items_global.extend(final_list)

    print(f"\n{'Swap re‑ranking' if not args.baseline_only else 'Retriever‑only baseline'} (keep={args.keep}, pool={args.candidate_pool}):")
    for k in range(1, args.topk+1):
        print(f"HR@{k}: {hits[k]/total:.4f}  NDCG@{k}: {ndcg[k]/total:.4f}")
    print(f"Average Log Popularity Difference: {np.mean(pop_diffs):.4f}")
    print(f"Average Intra‑list Diversity: {np.mean(diversities):.4f}")
    print(f"Catalogue Coverage: {len(set(recommended_items_global))/len(all_asins):.4f}")

if __name__ == '__main__':
    main()
