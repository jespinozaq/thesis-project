import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import faiss

from config import get_args
from data_utils import AmazonReviewDataset, collate_fn
from model import BLAIRRecommender
from losses import scaled_cross_entropy_loss, popularity_penalty, diversity_regularizer
from transformers import AutoTokenizer

def build_item_index(model, dataset, device, batch_size=256):
    """Precompute item embeddings and build FAISS index."""
    model.eval()
    item_embs = []
    item_asins = list(dataset.item_meta.keys())
    # Create a temporary DataLoader for item metadata
    class ItemDataset(torch.utils.data.Dataset):
        def __init__(self, asins, meta_dict, tokenizer, max_len):
            self.asins = asins
            self.meta_dict = meta_dict
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self): return len(self.asins)
        def __getitem__(self, idx):
            asin = self.asins[idx]
            meta = self.meta_dict[asin]
            enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors='pt')
            return {'input_ids': enc['input_ids'].squeeze(0),
                    'attention_mask': enc['attention_mask'].squeeze(0),
                    'asin': asin}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    item_ds = ItemDataset(item_asins, dataset.item_meta, tokenizer, args.max_seq_length)
    item_loader = DataLoader(item_ds, batch_size=batch_size, shuffle=False)

    all_embs = []
    all_asins = []
    with torch.no_grad():
        for batch in tqdm(item_loader, desc="Building item index"):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            emb = model.encode_item(input_ids, attn_mask)
            all_embs.append(emb.cpu().numpy())
            all_asins.extend(batch['asin'])
    all_embs = np.vstack(all_embs).astype('float32')
    index = faiss.IndexFlatIP(all_embs.shape[1])  # inner product (cosine since normalized)
    index.add(all_embs)
    return index, all_asins, all_embs

def retrieve_topk(model, context_embs, item_index, item_asins, item_logpop_dict, k=10):
    """Retrieve top-k items for each context embedding using FAISS."""
    D, I = item_index.search(context_embs.cpu().numpy(), k)  # I: [B, K]
    topk_logpops = []
    topk_embs = []
    for i in range(context_embs.size(0)):
        idxs = I[i]
        # get log popularity for each retrieved item
        logpops = [item_logpop_dict[item_asins[idx]] for idx in idxs]
        topk_logpops.append(logpops)
        # also retrieve embeddings for diversity loss
        embs = torch.tensor(item_index.reconstruct_n(idx, k)).to(context_embs.device)
        topk_embs.append(embs)
    topk_logpops = torch.tensor(topk_logpops, device=context_embs.device)  # [B, K]
    topk_embs = torch.stack(topk_embs, dim=0)  # [B, K, D]
    return topk_logpops, topk_embs

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load augmentation cache if used
    aug_dict = None
    if args.use_augmentation:
        import json
        with open(args.augmentation_cache, 'r') as f:
            aug_dict = json.load(f)

    # Dataset
    train_dataset = AmazonReviewDataset(
        args.review_file, args.meta_file, tokenizer,
        max_length=args.max_seq_length,
        augmentation_dict=aug_dict, use_augmentation=args.use_augmentation
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)

    # Model
    model = BLAIRRecommender(args.model_name).to(device)
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()

    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    # Item index for retrieval (rebuild periodically, but for simplicity build once)
    # common practice is to rebuild every epoch or few steps. 
    print("Building item index for retrieval...")
    item_index, item_asins, item_embs = build_item_index(model, train_dataset, device, batch_size=256)
    item_logpop_dict = train_dataset.item_logpop

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(pbar):
            # Move to device
            c_ids = batch['context_input_ids'].to(device)
            c_mask = batch['context_attention_mask'].to(device)
            m_ids = batch['meta_input_ids'].to(device)
            m_mask = batch['meta_attention_mask'].to(device)
            item_freqs = batch['item_freq'].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                c_emb = model.encode_context(c_ids, c_mask)
                m_emb = model.encode_item(m_ids, m_mask)

                # 1. SCE loss
                loss_sce = scaled_cross_entropy_loss(
                    c_emb, m_emb, item_freqs,
                    temperature=args.temperature, beta=args.beta
                )

                # 2. Popularity penalty (requires top-k retrieval)
                with torch.no_grad():
                    topk_logpops, topk_embs = retrieve_topk(
                        model, c_emb, item_index, item_asins, item_logpop_dict, k=args.topk
                    )
                # user_hist_logpop: average of items in batch (since batch random, approximate)
                # Here I'm  use the positive item's logpop as a proxy.
                user_hist_logpop = batch['item_logpop'].to(device)
                loss_pop = popularity_penalty(topk_logpops, user_hist_logpop)

                # 3. Diversity regularizer
                loss_div = diversity_regularizer(topk_embs)

                total_loss = loss_sce + args.lambda_pop * loss_pop + args.gamma_div * loss_div

            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                'loss_sce': f"{loss_sce.item():.4f}",
                'loss_pop': f"{loss_pop.item():.4f}",
                'loss_div': f"{loss_div.item():.4f}"
            })

        # Save checkpoint
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"model_epoch{epoch+1}.pt"))
        # Rebuild index after each epoch (embeddings changed)
        item_index, item_asins, item_embs = build_item_index(model, train_dataset, device, batch_size=256)

    print("Training complete.")

if __name__ == "__main__":
    args = get_args()
    train(args)