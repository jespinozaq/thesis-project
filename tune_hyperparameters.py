import os, json, gc, csv, itertools
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np, faiss
from collections import defaultdict

from data_utils2 import ProcessedPairDataset, collate_fn
from model import BLAIRRecommender
from losses import popularity_penalty, diversity_regularizer
from losses_framework import inverse_propensity_weighted_sce, augmentation_consistency_loss
import argparse

# ---------- Extended validation (HR, NDCG, LogPopDiff, Diversity) ----------
def build_item_index(model, meta_dict, device, batch_size, max_seq_length, model_name):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len):
            self.asins = asins; self.meta_dict = meta_dict; self.tokenizer = tokenizer; self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]; meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 'asin': asin}
    loader = DataLoader(ItemDataset(list(meta_dict.keys()), meta_dict, tokenizer, max_len=max_seq_length), batch_size=batch_size, shuffle=False)
    all_embs, all_asins = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building item index"):
            emb = model.encode_item(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            all_embs.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    all_embs = np.vstack(all_embs).astype('float32')
    cpu_index = faiss.IndexFlatIP(all_embs.shape[1])
    cpu_index.add(all_embs)
    return cpu_index, all_asins, all_embs

def retrieve_topk(context_embs, item_embs_gpu, item_asins, item_logpop_dict, k=10):
    sim = torch.matmul(context_embs, item_embs_gpu.t())
    _, topk_idx = torch.topk(sim, k, dim=-1)
    topk_logpops, topk_embs = [], []
    for i in range(context_embs.size(0)):
        idxs = topk_idx[i].tolist()
        logpops = [item_logpop_dict[item_asins[idx]] for idx in idxs]
        topk_logpops.append(logpops)
        embs = item_embs_gpu[idxs]
        topk_embs.append(embs)
    topk_logpops = torch.tensor(topk_logpops, device=context_embs.device)
    topk_embs = torch.stack(topk_embs, dim=0)
    return topk_logpops, topk_embs

def validate(model, val_seqs, meta_dict, item_embs, index_asins, faiss_index, item_logpop_dict, device, k=10):
    model.eval()
    total = len(val_seqs)
    hits = 0; ndcg_sum = 0.0; pop_diffs = []; diversities = []
    with torch.no_grad():
        for uid, data in val_seqs.items():
            history = data['history']; target_asin = data['target_asin']
            hist_embs = [item_embs[index_asins.index(a)] for a in history if a in index_asins]
            user_emb = np.mean(hist_embs, axis=0) if hist_embs else np.mean(item_embs, axis=0)
            norm = np.linalg.norm(user_emb)
            if norm > 0: user_emb /= norm
            D, I = faiss_index.search(np.expand_dims(user_emb.astype('float32'), 0), k)
            retrieved = [index_asins[i] for i in I[0]]
            if target_asin in retrieved:
                rank = retrieved.index(target_asin) + 1
                hits += 1; ndcg_sum += 1.0 / np.log2(rank + 1)
            rec_logpops = [item_logpop_dict[a] for a in retrieved]
            hist_logpops = [item_logpop_dict[a] for a in history if a in item_logpop_dict]
            pop_diffs.append(np.mean(rec_logpops) - (np.mean(hist_logpops) if hist_logpops else np.mean(rec_logpops)))
            retrieved_embs = np.array([item_embs[index_asins.index(a)] for a in retrieved])
            sim_matrix = np.dot(retrieved_embs, retrieved_embs.T)
            np.fill_diagonal(sim_matrix, 0)
            if len(retrieved_embs) > 1:
                offdiag = sim_matrix[~np.eye(len(retrieved_embs), dtype=bool)]
                avg_sim = offdiag.mean()
            else:
                avg_sim = 0.0
            diversities.append(1.0 - avg_sim)
    hr = hits / total; ndcg = ndcg_sum / total
    avg_popdiff = np.mean(pop_diffs)
    avg_diversity = np.mean(diversities)
    model.train()
    return hr, ndcg, avg_popdiff, avg_diversity

# ---------- Quick training for one combination (returns full metrics) ----------
def train_one_combo(args, lpop, gdiv, laug, seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed); np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)

    with open(args.meta_file, 'r', encoding='utf-8') as f:
        meta_list = [json.loads(line.strip()) for line in f]
    meta_dict = {}
    for item in meta_list:
        asin = item['parent_asin']
        parts = [item.get('title','')]
        if 'features' in item: parts.extend(item['features'])
        if 'description' in item: parts.extend(item['description'])
        meta_dict[asin] = ' '.join(parts).replace('\n',' ').strip()
    with open(args.item_pop_file, 'r', encoding='utf-8') as f:
        item_freq = json.load(f)

    aug_dict = {}
    if args.use_augmentation:
        with open(args.augmentation_cache, 'r', encoding='utf-8') as f:
            aug_dict = json.load(f)

    train_dataset = ProcessedPairDataset(pair_file=args.train_pairs_file, meta_dict=meta_dict, item_freq=item_freq, tokenizer=tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = BLAIRRecommender(args.model_name)
    model.context_encoder.transformer.load_state_dict(backbone.state_dict())
    model.item_encoder.transformer.load_state_dict(backbone.state_dict())
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) * args.tune_epochs) // max(1, args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=args.fp16)

    torch.cuda.empty_cache(); gc.collect()
    item_index, item_asins, item_embs = build_item_index(model, meta_dict, device, batch_size=128, max_seq_length=args.max_seq_length, model_name=args.model_name)
    torch.cuda.empty_cache(); gc.collect()
    item_embs_gpu = torch.from_numpy(item_embs).to(device)
    item_logpop_dict = train_dataset.item_logpop

    with open(args.val_seq, 'r') as f:
        val_seqs = json.load(f)

    best_score = -1e9
    best_hr = -1.0
    best_epoch = -1
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.tune_epochs):
        pbar = tqdm(train_loader, desc=f"[pop={lpop},div={gdiv},aug={laug}] Epoch {epoch+1}/{args.tune_epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            m_ids = batch['meta_input_ids'].to(device)
            m_mask = batch['meta_attention_mask'].to(device)
            item_freqs = batch['item_freq'].to(device)

            with autocast(enabled=args.fp16):
                c_emb = model.encode_context(c_ids, c_mask)
                m_emb = model.encode_item(m_ids, m_mask)
                loss_sce = inverse_propensity_weighted_sce(c_emb, m_emb, item_freqs, temperature=args.temperature, beta=args.beta)
                with torch.no_grad():
                    topk_logpops, topk_embs = retrieve_topk(c_emb, item_embs_gpu, item_asins, item_logpop_dict, k=args.topk)
                user_hist_logpop = batch['item_logpop'].to(device)
                loss_pop = popularity_penalty(topk_logpops, user_hist_logpop)
                loss_div = diversity_regularizer(topk_embs)
                total_loss = loss_sce + lpop * loss_pop + gdiv * loss_div
                if args.use_augmentation:
                    aug_strs = [aug_dict.get(a, meta_dict[a]) for a in batch['item_asin']]
                    aug_enc = tokenizer(aug_strs, truncation=True, padding='max_length', max_length=args.max_seq_length, return_tensors='pt')
                    aug_emb = model.encode_item(aug_enc['input_ids'].to(device), aug_enc['attention_mask'].to(device))
                    loss_aug = augmentation_consistency_loss(m_emb, aug_emb, temperature=0.07)
                    total_loss = total_loss + laug * loss_aug

            total_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(total_loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

        hr, ndcg, popdiff, div = validate(model, val_seqs, meta_dict, item_embs, item_asins, item_index, item_logpop_dict, device, k=10)
        score = hr + args.alpha * popdiff + args.beta * div   # note: popdiff is negative, so -popdiff is positive
        print(f"[pop={lpop},div={gdiv},aug={laug}] Epoch {epoch+1}: HR@10={hr:.4f}, PopDiff={popdiff:.4f}, Div={div:.4f}, Score={score:.4f}")
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_hr = hr
            best_popdiff = popdiff
            best_div = div

        torch.cuda.empty_cache(); gc.collect()
        item_index, item_asins, item_embs = build_item_index(model, meta_dict, device, batch_size=128, max_seq_length=args.max_seq_length, model_name=args.model_name)
        torch.cuda.empty_cache(); gc.collect()
        item_embs_gpu = torch.from_numpy(item_embs).to(device)

    return best_score, best_hr, best_popdiff, best_div, best_epoch

# ---------- Main ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pairs_file', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop_file', required=True)
    parser.add_argument('--val_seq', required=True)
    parser.add_argument('--model_name', default='blair-roberta-base-local')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--tune_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.2, help='SCE frequency scaling exponent (not to confuse with fairness weight)')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--use_augmentation', action='store_true', default=True)
    parser.add_argument('--augmentation_cache', default='augmented_descriptions_full.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_csv', default='hyperparameter_tuning_results.csv')
    parser.add_argument('--pop_vals', nargs='+', type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument('--div_vals', nargs='+', type=float, default=[0.05, 0.1])
    parser.add_argument('--aug_vals', nargs='+', type=float, default=[0.05, 0.1, 0.2])
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for -LogPopDiff in fairness score')
    parser.add_argument('--beta_fair', type=float, default=0.5, help='Weight for Diversity in fairness score')
    parser.add_argument('--run_full_best', action='store_true', help='Train full 20‑epoch model with best hyperparameters')
    args = parser.parse_args()

    best_score = -1e9
    best_params = None
    results = []

    for lpop, gdiv, laug in itertools.product(args.pop_vals, args.div_vals, args.aug_vals):
        print(f"\n{'='*60}\nTuning λ_pop={lpop}, γ_div={gdiv}, η_aug={laug}\n{'='*60}")
        trial_score, trial_hr, trial_popdiff, trial_div, trial_epoch = train_one_combo(args, lpop, gdiv, laug, seed=args.seed)
        results.append({
            'lambda_pop': lpop, 'gamma_div': gdiv, 'lambda_aug': laug,
            'best_score': round(trial_score, 4),
            'best_hr': round(trial_hr, 4),
            'best_popdiff': round(trial_popdiff, 4),
            'best_diversity': round(trial_div, 4),
            'best_epoch': trial_epoch
        })
        if trial_score > best_score:
            best_score = trial_score
            best_params = (lpop, gdiv, laug)

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lambda_pop','gamma_div','lambda_aug','best_score','best_hr','best_popdiff','best_diversity','best_epoch'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nTuning finished. Results saved to {args.output_csv}")
    print(f"Best combination: λ_pop={best_params[0]}, γ_div={best_params[1]}, η_aug={best_params[2]} (Score={best_score:.4f})")
    if args.run_full_best:
        print("\nLaunching full training with best hyperparameters...")
        import subprocess
        cmd = ['python', 'train_framework.py',
               '--train_pairs_file', args.train_pairs_file,
               '--meta_file', args.meta_file,
               '--item_pop_file', args.item_pop_file,
               '--use_augmentation', '--augmentation_cache', args.augmentation_cache,
               '--lambda_pop', str(best_params[0]), '--gamma_div', str(best_params[1]), '--lambda_aug', str(best_params[2]),
               '--output_dir', 'checkpoints_blair_base_fullaug_best_tuned',
               '--model_name', args.model_name,
               '--batch_size', str(args.batch_size),
               '--gradient_accumulation_steps', str(args.gradient_accumulation_steps),
               '--epochs', '20', '--cpt_epochs', '2', '--lr', str(args.lr),
               '--temperature', str(args.temperature), '--beta', str(args.beta),
               '--topk', str(args.topk), '--fp16', '--val_seq', args.val_seq,
               '--max_seq_length', str(args.max_seq_length), '--seed', str(args.seed)]
        subprocess.run(cmd)
