import json, random, sys, argparse

def load_meta(meta_path):
    """Return dict asin -> short description (title + first 200 chars of metadata)."""
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            asin = item['parent_asin']
            parts = [item.get('title', '')]
            if 'features' in item: parts.extend(item['features'])
            if 'description' in item: parts.extend(item['description'])
            meta[asin] = ' '.join(parts).replace('\n', ' ').strip()[:200]   # first 200 chars
    return meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', default='meta_All_Beauty.jsonl',
                        help='Path to metadata file (default: meta_All_Beauty.jsonl)')
    args_cli = parser.parse_args()

    models = {
        "Custom Loss (GPT-4o-mini)": "recommendations_base_fullaug.json",
        "Custom Loss (DeepSeek)": "recommendations_base_deepseek_fullaug.json",
        "Custom Loss (GPT-4o)": "recommendations_base_gpt4o_fullaug.json",
        "LLM-as-Embedder": "recommendations_llm_fullaug.json"
    }

    meta_dict = load_meta(args_cli.meta_file)
    recs = {}
    for name, path in models.items():
        try:
            recs[name] = json.load(open(path))
        except FileNotFoundError:
            print(f"Warning: {path} not found, skipping {name}")

    if not recs:
        sys.exit("No recommendation files found.")

    common = set.intersection(*[set(r.keys()) for r in recs.values()])
    sample = random.sample(sorted(common), min(5, len(common)))
    print("Qualitative examples:\n")
    for uid in sample:
        history = recs[list(recs.keys())[0]][uid]['history']
        print(f"User: {uid}")
        print("  History items:")
        for asin in history[:5]:
            print(f"    - {meta_dict.get(asin, asin)[:180]}")
        print()
        for model_name, data in recs.items():
            top10 = data[uid]['recommended']
            print(f"  {model_name}:")
            for i, asin in enumerate(top10, 1):
                print(f"    {i:2d}. {meta_dict.get(asin, asin)[:180]}")
            print()
        print("-" * 80)

if __name__ == '__main__':
    main()
