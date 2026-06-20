import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np

class AmazonReviewDataset(Dataset):
    """
    Creates training pairs (context, item_metadata) from reviews.
    Also computes item popularity statistics.
    """
    def __init__(self, review_file, meta_file, tokenizer, max_length=64,
                 augmentation_dict=None, use_augmentation=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = []
        self.item_freq = defaultdict(int)       # raw count of reviews per item
        self.item_meta = {}                     # asin -> metadata string
        self.augmentation_dict = augmentation_dict or {}

        # Load metadata first
        print("Loading item metadata...")
        with open(meta_file, 'r') as f:
            for line in f:
                meta = json.loads(line.strip())
                asin = meta['parent_asin']
                # Build metadata string: title + features + description
                meta_parts = [meta.get('title', '')]
                if 'features' in meta:
                    meta_parts.extend(meta['features'])
                if 'description' in meta:
                    meta_parts.extend(meta['description'])
                meta_str = ' '.join(meta_parts).replace('\n', ' ').strip()
                self.item_meta[asin] = meta_str
                self.item_freq[asin] = 0   # will count from reviews

        # Load reviews and create pairs
        print("Loading reviews and creating pairs...")
        with open(review_file, 'r') as f:
            for line in f:
                review = json.loads(line.strip())
                asin = review['parent_asin']
                if asin not in self.item_meta:
                    continue
                self.item_freq[asin] += 1

                # Context: review title + text
                context = f"{review.get('title', '')} {review.get('text', '')}".strip()
                # Item metadata (maybe augmented)
                if use_augmentation and asin in self.augmentation_dict:
                    meta_str = self.augmentation_dict[asin]
                else:
                    meta_str = self.item_meta[asin]

                if len(context) < 30 or len(meta_str) < 30:
                    continue

                self.pairs.append((context, meta_str, asin))

        # Compute log popularity for each item (add 1 smoothing)
        self.item_logpop = {asin: np.log(count + 1) for asin, count in self.item_freq.items()}
        print(f"Dataset created: {len(self.pairs)} pairs, {len(self.item_meta)} items.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, meta, asin = self.pairs[idx]
        context_enc = self.tokenizer(context, truncation=True, padding='max_length',
                                     max_length=self.max_length, return_tensors='pt')
        meta_enc = self.tokenizer(meta, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        return {
            'context_input_ids': context_enc['input_ids'].squeeze(0),
            'context_attention_mask': context_enc['attention_mask'].squeeze(0),
            'meta_input_ids': meta_enc['input_ids'].squeeze(0),
            'meta_attention_mask': meta_enc['attention_mask'].squeeze(0),
            'item_asin': asin,
            'item_freq': self.item_freq[asin],
            'item_logpop': self.item_logpop[asin]
        }

def collate_fn(batch):
    """Custom collate to handle variable keys."""
    keys = batch[0].keys()
    collated = {}
    for key in keys:
        if key in ['item_asin']:
            collated[key] = [b[key] for b in batch]
        elif key in ['item_freq', 'item_logpop']:
            collated[key] = torch.tensor([b[key] for b in batch], dtype=torch.float)
        else:
            collated[key] = torch.stack([b[key] for b in batch])
    return collated