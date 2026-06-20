import argparse
import json
from collections import defaultdict
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file', required=True)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--output_dir', default='./processed')
    return parser.parse_args()

def load_meta(meta_path):
    """Return dict asin -> metadata string (title + features + description)."""
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item:
                parts.extend(item['features'])
            if 'description' in item:
                parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n', ' ').strip()
    return meta

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading metadata...")
    meta_dict = load_meta(args.meta_file)

    # Group reviews by user
    print("Loading reviews...")
    user_items = defaultdict(list)   # user_id -> list of (timestamp, asin, review_text)
    with open(args.review_file, 'r') as f:
        for line in f:
            rev = json.loads(line.strip())
            uid = rev['user_id']
            asin = rev['parent_asin']
            if asin not in meta_dict:
                continue
            ts = rev['timestamp']
            context = f"{rev.get('title', '')} {rev.get('text', '')}".strip()
            if len(context) < 30 or len(meta_dict[asin]) < 30:
                continue
            user_items[uid].append((ts, asin, context))

    # Sort each user's interactions and split
    train_pairs = []
    val_seqs = {}
    test_seqs = {}

    # For popularity metrics, count total interactions per item (across all users)
    item_popularity = defaultdict(int)

    for uid, interactions in user_items.items():
        # Sort by timestamp ascending
        interactions.sort(key=lambda x: x[0])
        # Count item occurrences
        for _, asin, _ in interactions:
            item_popularity[asin] += 1

        n = len(interactions)
        if n < 3:
            # Skip users with too few interactions
            continue

        # Last one -> test, second last -> validation, rest -> train
        test_item = interactions[-1]
        val_item = interactions[-2]
        train_items = interactions[:-2]

        # Save training pairs: each (context, item_asin, metadata)
        for _, asin, context in train_items:
            train_pairs.append({'context': context, 'asin': asin})

        # Store evaluation sequences: history is the list of asins before the target
        # For test user, history includes all training + validation asins
        history_test = [asin for _, asin, _ in interactions[:-1]]   # all except test
        test_seqs[uid] = {
            'history': history_test,
            'target_asin': test_item[1]
        }

        # For validation, history up to val (i.e., training items only)
        history_val = [asin for _, asin, _ in train_items]
        val_seqs[uid] = {
            'history': history_val,
            'target_asin': val_item[1]
        }

    # Write training pairs as JSONL (compatible with existing AmazonReviewDataset logic)
    train_pairs_path = os.path.join(args.output_dir, 'train_pairs.jsonl')
    with open(train_pairs_path, 'w') as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + '\n')

    # Write val and test sequences as JSON
    val_path = os.path.join(args.output_dir, 'val_seqs.json')
    with open(val_path, 'w') as f:
        json.dump(val_seqs, f)

    test_path = os.path.join(args.output_dir, 'test_seqs.json')
    with open(test_path, 'w') as f:
        json.dump(test_seqs, f)

    # Write item popularity stats (needed for log popularity metrics)
    stats_path = os.path.join(args.output_dir, 'item_popularity.json')
    with open(stats_path, 'w') as f:
        json.dump(item_popularity, f)

    print(f"Saved {len(train_pairs)} training pairs to {train_pairs_path}")
    print(f"Validation users: {len(val_seqs)}, Test users: {len(test_seqs)}")
    print("Preprocessing done.")

if __name__ == '__main__':
    main()