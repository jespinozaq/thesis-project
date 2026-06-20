import os, json, gc, csv, subprocess, re
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np, faiss
from collections import defaultdict
import argparse

from data_utils2 import ProcessedPairDataset, collate_fn
from model import BLAIRRecommender

# ─── Loss Functions ──────────────────────────────────────────────────

def info_nce_loss(context_emb, item_emb, temperature=0.05):
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(sim, labels)

def bpr_loss(context_emb, item_emb, num_negatives=1, temperature=0.05):
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    pos_scores = torch.diag(sim)
    total_loss = 0.0
    for i in range(B):
        neg_mask = torch.ones(B, device=sim.device).bool()
        neg_mask[i] = False
        neg_scores = sim[i][neg_mask]
        diff = pos_scores[i].unsqueeze(0) - neg_scores
        total_loss += -torch.nn.functional.logsigmoid(diff).mean()
    return total_loss / B

def debiased_info_nce_loss(context_emb, item_emb, item_freqs, temperature=0.05, tau_plus=0.1):
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    max_freq = item_freqs.max()
    prob = (item_freqs.float() / (max_freq + 1e-8)).unsqueeze(0)
    debias = torch.log(prob + 1e-8)
    sim_debiased = sim - tau_plus * debias
    labels = torch.arange(B, device=sim.device)
    return torch.nn.functional.cross_entropy(sim_debiased, labels)

def bc_loss(context_emb, item_emb, item_freqs, temperature=0.05, margin_base=0.5):
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    max_freq = item_freqs.max()
    margins = margin_base * (item_freqs.float() / (max_freq + 1e-8)).unsqueeze(0)
    margin_matrix = margins.expand(B, B)
    eye = torch.eye(B, device=sim.device)
    margin_matrix = margin_matrix * (1 - eye)
    sim_margined = sim - margin_matrix
    labels = torch.arange(B, device=sim.device)
    return torch.nn.functional.cross_entropy(sim_margined, labels)

def sce_loss(context_emb, item_emb, item_freqs, temperature=0.05, beta=0.2):
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    max_freq = item_freqs.max()
    weights = (item_freqs / max_freq).pow(beta).unsqueeze(0)
    mask = torch.eye(B, device=sim.device).bool()
    neg_weights = weights.expand(B, B).clone()
    neg_weights[mask] = 1.0
    sim_scaled = sim - torch.log(neg_weights + 1e-8)
    labels = torch.arange(B, device=sim.device)
    return torch.nn.functional.cross_entropy(sim_scaled, labels)

def ipw_sce_loss(context_emb, item_emb, item_freqs, temperature=0.05, beta=0.2):
    B = context_emb.size(0)
    sim = torch.matmul(context_emb, item_emb.T) / temperature
    eps = 1e-8
    w = 1.0 / (item_freqs.float() + eps)
    w = w / w.sum() * B
    max_freq = item_freqs.max()
    freq_scale = (item_freqs.float() / (max_freq + eps)).pow(beta).unsqueeze(0)
    mask = torch.eye(B, device=sim.device).bool()
    neg_weights = freq_scale.expand(B, B).clone()
    neg_weights[mask] = 1.0
    sim_scaled = sim - torch.log(neg_weights + eps)
    labels = torch.arange(B, device=sim.device)
    ce = torch.nn.functional.cross_entropy(sim_scaled, labels, reduction='none')
    return (ce * w).mean()

LOSS_FUNCTIONS = {
    'infonce':  lambda c, m, f: info_nce_loss(c, m, temperature=0.05),
    'bpr':      lambda c, m, f: bpr_loss(c, m, temperature=0.05),
    'debiased': lambda c, m, f: debiased_info_nce_loss(c, m, f, temperature=0.05),
    'bc':       lambda c, m, f: bc_loss(c, m, f, temperature=0.05),
    'sce':      lambda c, m, f: sce_loss(c, m, f, temperature=0.05, beta=0.2),
    'ipw_sce':  lambda c, m, f: ipw_sce_loss(c, m, f, temperature=0.05, beta=0.2),
}

# ─── Item index & validation (with fairness metrics) ─────────────────

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

# ─── Training one loss variant ───────────────────────────────────────

def train_one_loss(loss_name, loss_fn, args):
    import os
    output_dir = os.path.join(args.output_dir, loss_name)
    if os.path.isfile(os.path.join(output_dir, "best_model.pt")):
        print(f"[{loss_name}] already completed, skipping.")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    val_seqs = None
    if args.val_seq and os.path.exists(args.val_seq):
        with open(args.val_seq, 'r') as f: val_seqs = json.load(f)
        print(f"Loaded {len(val_seqs)} validation users.")
    item_logpop_dict = {asin: np.log(item_freq.get(asin,0)+1) for asin in meta_dict}

    train_dataset = ProcessedPairDataset(pair_file=args.train_pairs_file, meta_dict=meta_dict, item_freq=item_freq, tokenizer=tokenizer, max_length=args.max_seq_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)

    model = BLAIRRecommender(args.model_name)
    model.context_encoder.transformer.load_state_dict(backbone.state_dict())
    model.item_encoder.transformer.load_state_dict(backbone.state_dict())
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = (len(train_loader) * args.epochs) // max(1, args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler(enabled=args.fp16)

    torch.cuda.empty_cache(); gc.collect()
    item_index, item_asins, item_embs = build_item_index(model, meta_dict, device, batch_size=128, max_seq_length=args.max_seq_length, model_name=args.model_name)
    torch.cuda.empty_cache(); gc.collect()

    best_metric = -1.0; patience = 10; no_improve = 0
    best_epoch = -1
    os.makedirs(output_dir, exist_ok=True)

    model.train(); global_step = 0; optimizer.zero_grad()
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"[{loss_name}] Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            m_ids = batch['meta_input_ids'].to(device)
            m_mask = batch['meta_attention_mask'].to(device)
            item_freqs = batch['item_freq'].to(device)

            with autocast(enabled=args.fp16):
                c_emb = model.encode_context(c_ids, c_mask)
                m_emb = model.encode_item(m_ids, m_mask)
                loss = loss_fn(c_emb, m_emb, item_freqs)

            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer); scaler.update()
                scheduler.step(); optimizer.zero_grad()
                global_step += 1
            pbar.set_postfix({'loss': loss.item()})

        if val_seqs is not None:
            hr, ndcg, popdiff, div = validate(model, val_seqs, meta_dict, item_embs, item_asins, item_index, item_logpop_dict, device, k=10)
            print(f"[{loss_name}] Epoch {epoch+1}: HR@10={hr:.4f}  NDCG@10={ndcg:.4f}  PopDiff={popdiff:.4f}  Diversity={div:.4f}")
            if hr > best_metric:
                best_metric = hr; no_improve = 0; best_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
                print(f"  >> New best (HR@10: {best_metric:.4f})")
                best_popdiff = popdiff; best_diversity = div; best_ndcg = ndcg
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}"); break

        torch.cuda.empty_cache(); gc.collect()
        item_index, item_asins, item_embs = build_item_index(model, meta_dict, device, batch_size=128, max_seq_length=args.max_seq_length, model_name=args.model_name)
        torch.cuda.empty_cache(); gc.collect()

    print(f"[{loss_name}] Training complete. Best epoch: {best_epoch}, HR@10={best_metric:.4f}, NDCG@10={best_ndcg:.4f}, PopDiff={best_popdiff:.4f}, Diversity={best_diversity:.4f}")
    return {
        'loss': loss_name,
        'best_epoch': best_epoch,
        'val_hr': best_metric,
        'val_ndcg': best_ndcg,
        'val_popdiff': best_popdiff,
        'val_diversity': best_diversity,
        'checkpoint': os.path.join(output_dir, "best_model.pt")
    }

# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pairs_file', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--item_pop_file', required=True)
    parser.add_argument('--val_seq', required=True)
    parser.add_argument('--output_dir', default='checkpoints_loss_comparison')
    parser.add_argument('--model_name', default='blair-roberta-base-local')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true', default=True)
    parser.add_argument('--losses', nargs='+', default=['infonce','bpr','debiased','bc','sce','ipw_sce'])
    parser.add_argument('--run_full_eval', action='store_true', help='Run full evaluation on top-3 loss checkpoints')
    parser.add_argument('--eval_script', type=str, default='evaluate_finetuned.py')
    parser.add_argument('--test_seq', type=str, default='processed_All_Beauty/test_seqs.json')
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    args = parser.parse_args()

    results = []
    for loss_name in args.losses:
        if loss_name not in LOSS_FUNCTIONS:
            print(f"Unknown loss '{loss_name}', skipping.")
            continue
        print(f"\n{'='*60}\nTraining with {loss_name.upper()}\n{'='*60}")
        res = train_one_loss(loss_name, LOSS_FUNCTIONS[loss_name], args)
        if res is not None:
            # remove checkpoint before writing CSV
            csv_row = {k:v for k,v in res.items() if k != 'checkpoint'}
            results.append(csv_row)

    # Save validation results to CSV
    csv_path = 'loss_comparison.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['loss', 'best_epoch', 'val_hr', 'val_ndcg', 'val_popdiff', 'val_diversity'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Validation comparison saved to {csv_path}")

    # Optional: run full evaluation on top 3 (by val HR@10)
    if args.run_full_eval:
        # Re‑attach checkpoint paths (they were removed to avoid CSV conflicts)
        for row in results:
            row['checkpoint'] = os.path.join(args.output_dir, row['loss'], "best_model.pt")
        top3 = sorted(results, key=lambda x: x['val_hr'], reverse=True)[:3]
        print("Running full evaluation on top 3 losses...")
        test_metrics = []
        for res in top3:
            cmd = [
                'python', args.eval_script,
                '--model_path', res['checkpoint'],
                '--model_name', args.model_name,
                '--test_seq', args.test_seq,
                '--meta_file', args.meta_file,
                '--item_pop', args.item_pop_file,
                '--topk', str(args.topk),
                '--batch_size', str(args.eval_batch_size)
            ]
            print(f"Evaluating {res['loss']}...")
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            out = proc.stdout + "\n" + proc.stderr
            hr = ndcg = popdiff = diversity = coverage = None
            for line in out.splitlines():
                m = re.search(r'HR@10:\s+([0-9.]+)', line)
                if m: hr = float(m.group(1))
                m = re.search(r'NDCG@10:\s+([0-9.]+)', line)
                if m: ndcg = float(m.group(1))
                m = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', line)
                if m: popdiff = float(m.group(1))
                m = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', line)
                if m: diversity = float(m.group(1))
                m = re.search(r'Catalogue Coverage:\s+([0-9.]+)', line)
                if m: coverage = float(m.group(1))
            test_metrics.append({
                'loss': res['loss'],
                'test_hr@10': hr,
                'test_ndcg@10': ndcg,
                'test_popdiff': popdiff,
                'test_diversity': diversity,
                'test_coverage': coverage
            })
        test_csv_path = 'loss_comparison_test.csv'
        with open(test_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['loss','test_hr@10','test_ndcg@10','test_popdiff','test_diversity','test_coverage'])
            writer.writeheader()
            writer.writerows(test_metrics)
        print(f"Test evaluation saved to {test_csv_path}")
