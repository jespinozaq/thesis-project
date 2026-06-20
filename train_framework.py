import os, json, gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np, faiss
from collections import defaultdict

from config import get_args
from data_utils2 import ProcessedPairDataset, collate_fn
from model import BLAIRRecommender
from losses import popularity_penalty, diversity_regularizer
from losses_framework import inverse_propensity_weighted_sce, augmentation_consistency_loss


def build_item_index(model, meta_dict, device, batch_size, max_seq_length, model_name):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    loader = DataLoader(ItemDataset(list(meta_dict.keys()), meta_dict, tokenizer, max_len=max_seq_length),
                        batch_size=batch_size, shuffle=False)
    all_embs, all_asins = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Building item index"):
            emb = model.encode_item(batch['input_ids'].to(device),
                                    batch['attention_mask'].to(device))
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


def validate(model, val_seqs, meta_dict, item_embs, index_asins, faiss_index, device, k=10):
    model.eval()
    total = len(val_seqs)
    hits = 0; ndcg_sum = 0.0
    with torch.no_grad():
        for uid, data in val_seqs.items():
            history = data['history']; target_asin = data['target_asin']
            hist_embs = []
            for a in history:
                if a in index_asins:
                    idx = index_asins.index(a)
                    hist_embs.append(item_embs[idx])
            if hist_embs:
                user_emb = np.mean(hist_embs, axis=0)
            else:
                user_emb = np.mean(item_embs, axis=0)
            norm = np.linalg.norm(user_emb)
            if norm > 0: user_emb /= norm
            D, I = faiss_index.search(np.expand_dims(user_emb.astype('float32'), 0), k)
            retrieved = [index_asins[i] for i in I[0]]
            if target_asin in retrieved:
                rank = retrieved.index(target_asin) + 1
                hits += 1
                ndcg_sum += 1.0 / np.log2(rank + 1)
    hr = hits / total; ndcg = ndcg_sum / total
    model.train()
    return hr, ndcg


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load tokenizer & BLAIR backbone ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    backbone = AutoModel.from_pretrained(args.model_name)

    # --- Metadata & popularity ---
    with open(args.meta_file, 'r', encoding='utf-8') as f:
        meta_list = [json.loads(line.strip()) for line in f]
    meta_dict = {}
    for item in meta_list:
        asin = item['parent_asin']
        parts = [item.get('title', '')]
        if 'features' in item: parts.extend(item['features'])
        if 'description' in item: parts.extend(item['description'])
        meta_dict[asin] = ' '.join(parts).replace('\n', ' ').strip()

    with open(args.item_pop_file, 'r', encoding='utf-8') as f:
        item_freq = json.load(f)

    # --- Validation sequences ---
    val_seqs = None
    if args.val_seq and os.path.exists(args.val_seq):
        with open(args.val_seq, 'r') as f:
            val_seqs = json.load(f)
        print(f"Loaded {len(val_seqs)} validation users.")
    else:
        print("No validation file – training for fixed epochs without early stopping.")

    # --- Augmentation cache ---
    aug_dict = {}
    if args.use_augmentation:
        with open(args.augmentation_cache, 'r', encoding='utf-8') as f:
            aug_dict = json.load(f)
        print(f"Loaded {len(aug_dict)} augmented descriptions.")

    # --- Dataset ---
    train_dataset = ProcessedPairDataset(
        pair_file=args.train_pairs_file, meta_dict=meta_dict,
        item_freq=item_freq, tokenizer=tokenizer, max_length=args.max_seq_length
    )
    # Inverse-popularity weights for CPT
    print("Computing inverse-popularity pair weights for CPT...")
    pair_weights = []
    for ctx, asin in train_dataset.pairs:
        freq = item_freq.get(asin, 1)
        pair_weights.append(1.0 / max(freq, 1))
    sampler = WeightedRandomSampler(pair_weights, num_samples=len(pair_weights), replacement=True)
    print(f"CPT sampler ready: {len(pair_weights)} pairs with weights min={min(pair_weights):.6f} max={max(pair_weights):.2f}")

    # --- Model ---
    model = BLAIRRecommender(args.model_name)
    model.context_encoder.transformer.load_state_dict(backbone.state_dict())
    model.item_encoder.transformer.load_state_dict(backbone.state_dict())
    model.to(device)

    # --- Optimiser & scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps_cpt = (len(train_dataset) * max(1, args.cpt_epochs)) // max(1, args.gradient_accumulation_steps)
    total_steps_sft = (len(train_dataset) * max(1, args.epochs - args.cpt_epochs)) // max(1, args.gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps_cpt + total_steps_sft)
    scaler = GradScaler(enabled=args.fp16)

    # --- Item index (CPU FAISS) + GPU embeddings ---
    torch.cuda.empty_cache(); gc.collect()
    item_index, item_asins, item_embs = build_item_index(
        model, meta_dict, device, batch_size=128, max_seq_length=args.max_seq_length,
        model_name=args.model_name)
    torch.cuda.empty_cache(); gc.collect()
    item_embs_gpu = torch.from_numpy(item_embs).to(device)
    item_logpop_dict = train_dataset.item_logpop

    # --- Early stopping ---
    best_metric = -1.0
    patience = 10

    no_improve = 0
    global_step = 0
    optimizer.zero_grad()

    # ============================================================
    #  Stage 1: Continued Pre‑Training (CPT)
    #  IPW‑SCE only, with inverse‑popularity weighted sampling
    # ============================================================
    cpt_epochs = max(1, args.cpt_epochs)
    print(f"\n=== CPT Stage: {cpt_epochs} epoch(s) with IPW‑SCE (inverse‑popularity sampling) ===\n")
    cpt_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            sampler=sampler, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)

    for epoch in range(cpt_epochs):
        pbar = tqdm(cpt_loader, desc=f"CPT Epoch {epoch+1}/{cpt_epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            m_ids = batch['meta_input_ids'].to(device)
            m_mask = batch['meta_attention_mask'].to(device)
            item_freqs = batch['item_freq'].to(device)

            with autocast(enabled=args.fp16):
                c_emb = model.encode_context(c_ids, c_mask)
                m_emb = model.encode_item(m_ids, m_mask)
                loss = inverse_propensity_weighted_sce(c_emb, m_emb, item_freqs,
                                                       temperature=args.temperature, beta=args.beta)

            loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            pbar.set_postfix({'loss': loss.item()})

        # Validation after each CPT epoch
        if val_seqs is not None:
            hr, ndcg = validate(model, val_seqs, meta_dict, item_embs,
                                item_asins, item_index, device, k=10)
            print(f"CPT Epoch {epoch+1} → Validation HR@10: {hr:.4f}  NDCG@10: {ndcg:.4f}")

        # Rebuild index after CPT epoch
        torch.cuda.empty_cache(); gc.collect()
        item_index, item_asins, item_embs = build_item_index(
            model, meta_dict, device, batch_size=128,
            max_seq_length=args.max_seq_length, model_name=args.model_name)
        torch.cuda.empty_cache(); gc.collect()
        item_embs_gpu = torch.from_numpy(item_embs).to(device)

    # Save CPT checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "cpt_checkpoint.pt"))
    print("CPT stage completed. Checkpoint saved.\n")

    # ============================================================
    #  Stage 2: Supervised Fine‑Tuning (SFT)
    #  Full multi‑objective loss with standard random shuffling
    # ============================================================
    sft_epochs = max(1, args.epochs - cpt_epochs)
    if sft_epochs > 0:
        print(f"\n=== SFT Stage: {sft_epochs} epoch(s) with full multi‑objective loss ===\n")

    # SFT uses standard random shuffling (not weighted sampler)
    sft_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)

    for epoch in range(sft_epochs):
        pbar = tqdm(sft_loader, desc=f"SFT Epoch {epoch+1}/{sft_epochs}")
        for step, batch in enumerate(pbar):
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            m_ids = batch['meta_input_ids'].to(device)
            m_mask = batch['meta_attention_mask'].to(device)
            item_freqs = batch['item_freq'].to(device)

            with autocast(enabled=args.fp16):
                c_emb = model.encode_context(c_ids, c_mask)
                m_emb = model.encode_item(m_ids, m_mask)

                loss_sce = inverse_propensity_weighted_sce(
                    c_emb, m_emb, item_freqs,
                    temperature=args.temperature, beta=args.beta)

                with torch.no_grad():
                    topk_logpops, topk_embs = retrieve_topk(
                        c_emb, item_embs_gpu, item_asins,
                        item_logpop_dict, k=args.topk)

                user_hist_logpop = batch['item_logpop'].to(device)
                loss_pop = popularity_penalty(topk_logpops, user_hist_logpop)
                loss_div = diversity_regularizer(topk_embs)

                total_loss = loss_sce + args.lambda_pop * loss_pop + args.gamma_div * loss_div

                if args.use_augmentation:
                    aug_strs = [aug_dict.get(a, meta_dict[a]) for a in batch['item_asin']]
                    aug_enc = tokenizer(aug_strs, truncation=True, padding='max_length',
                                        max_length=args.max_seq_length, return_tensors='pt')
                    aug_emb = model.encode_item(aug_enc['input_ids'].to(device),
                                                aug_enc['attention_mask'].to(device))
                    loss_aug = augmentation_consistency_loss(m_emb, aug_emb, temperature=0.07)
                    total_loss = total_loss + args.lambda_aug * loss_aug

            total_loss = total_loss / args.gradient_accumulation_steps
            scaler.scale(total_loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            log_dict = {'loss_sce': loss_sce.item(), 'loss_pop': loss_pop.item(),
                        'loss_div': loss_div.item()}
            if args.use_augmentation: log_dict['loss_aug'] = loss_aug.item()
            pbar.set_postfix(log_dict)

        # --- Validation & Early Stopping ---
        if val_seqs is not None:
            hr, ndcg = validate(model, val_seqs, meta_dict, item_embs,
                                item_asins, item_index, device, k=10)
            print(f"Validation HR@10: {hr:.4f}  NDCG@10: {ndcg:.4f}")
            current_metric = hr
            if current_metric > best_metric:
                best_metric = current_metric
                no_improve = 0
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                print(f"  >> New best model saved (HR@10: {best_metric:.4f})")
            else:
                no_improve += 1
                print(f"  >> No improvement for {no_improve} epoch(s)")
            if no_improve >= patience:
                print(f"Early stopping triggered after {cpt_epochs + epoch + 1} total epochs!")
                break

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch+1}.pt"))

        # Rebuild index after each SFT epoch
        torch.cuda.empty_cache(); gc.collect()
        item_index, item_asins, item_embs = build_item_index(
            model, meta_dict, device, batch_size=128,
            max_seq_length=args.max_seq_length, model_name=args.model_name)
        torch.cuda.empty_cache(); gc.collect()
        item_embs_gpu = torch.from_numpy(item_embs).to(device)

    print("Training complete. Best validation HR@10: {:.4f}".format(best_metric))


if __name__ == "__main__":
    args = get_args()
    train(args)
