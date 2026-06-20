import json, random, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', default='meta_All_Beauty.jsonl')
    parser.add_argument('--aug_cache', default='augmented_descriptions_full.json')
    parser.add_argument('--num_items', type=int, default=5)
    args = parser.parse_args()

    # Load original descriptions
    original = {}
    with open(args.meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            original[asin] = ' '.join(parts).replace('\n', ' ').strip()

    # Load augmented descriptions
    with open(args.aug_cache, 'r', encoding='utf-8') as f:
        augmented = json.load(f)

    # Select items that exist in both
    common = sorted(set(original) & set(augmented))
    sample = random.sample(common, min(args.num_items, len(common)))

    print("ORIGINAL vs AUGMENTED DESCRIPTIONS\n")
    for asin in sample:
        print(f"ASIN: {asin}")
        print(f"Original:\n{original[asin]}\n")
        print(f"Augmented:\n{augmented[asin]}\n")
        print("-" * 80)

if __name__ == '__main__':
    main()
