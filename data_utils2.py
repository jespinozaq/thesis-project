import json
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict

class ProcessedPairDataset(Dataset):
    """
    Loads training pairs (context, item_metadata) from a preprocessed JSONL file.
    Requires pre-built dictionaries: meta_dict (asin -> metadata str), item_freq (asin -> count).
    Optionally, a tokenizer and max_length for encoding.
    """
    def __init__(self, pair_file, meta_dict, item_freq, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.meta_dict = meta_dict
        self.item_freq = item_freq
        self.pairs = []

        # Load pairs: each line is {"context": "...", "asin": "..."}
        with open(pair_file, 'r') as f:
            for line in f:
                pair = json.loads(line.strip())
                asin = pair['asin']
                if asin not in meta_dict:
                    continue
                context = pair['context']
                if len(context) < 30 or len(meta_dict[asin]) < 30:
                    continue
                self.pairs.append((context, asin))

        # Compute log popularity for each item (add 1 smoothing)
        self.item_logpop = {asin: np.log(self.item_freq.get(asin, 0) + 1) for asin in self.meta_dict}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, asin = self.pairs[idx]
        meta_str = self.meta_dict[asin]
        context_enc = self.tokenizer(context, truncation=True, padding='max_length',
                                     max_length=self.max_length, return_tensors='pt')
        meta_enc = self.tokenizer(meta_str, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        return {
            'context_input_ids': context_enc['input_ids'].squeeze(0),
            'context_attention_mask': context_enc['attention_mask'].squeeze(0),
            'meta_input_ids': meta_enc['input_ids'].squeeze(0),
            'meta_attention_mask': meta_enc['attention_mask'].squeeze(0),
            'item_asin': asin,
            'item_freq': self.item_freq.get(asin, 0),
            'item_logpop': self.item_logpop.get(asin, 0.0)
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