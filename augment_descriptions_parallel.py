import json, argparse, time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def build_prompt_A(original_desc):
    # The prompt explicitly forbids promotional language and requires strict factual neutrality.
    # It includes a concrete example to guide the LLM.
    return (
        f"The description of an item is as follows: '{original_desc}'.\n\n"
        "To ensure this recommendation is fair and appealing to a diverse audience, generate an augmented "
        "description that:\n"
        "1. Uses **neutral, factual language** only. Do not add subjective praise, adjectives like "
        "'delightful' or 'exceptional', or unverifiable claims of quality.\n"
        "2. Restructures the original factual information to be inclusive of people of all ages, genders, "
        "and cultural backgrounds without adding new information.\n"
        "3. If the original description contains any language that might reflect or reinforce biases, "
        "rewrite it in a neutral, inclusive manner while preserving all factual details.\n"
        "4. If the original description is very short, you may rephrase the existing facts but do not "
        "invent new product characteristics.\n\n"
        "Example:\n"
        "Original: \"Beautiful necklace for women, best-seller.\"\n"
        "Augmented: \"Necklace made of [material], suitable for any person.\"\n\n"
        "Return only the plain, factual augmented description, without any additional text or commentary."
    )

def process_item(asin, desc, model, api_key, base_url=None):
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    prompt = build_prompt_A(desc)
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300          # increased to avoid truncation of longer descriptions
            )
            return asin, resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"Attempt {attempt} for {asin}: {e}")
            if attempt == max_attempts:
                print(f"Failed after {max_attempts} attempts, using original description.")
                return asin, desc
            time.sleep(2 ** attempt)

def load_meta(meta_path):
    meta = {}
    with open(meta_path, 'r', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--output_cache', required=True)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--api_key', required=True)
    parser.add_argument('--base_url', type=str, default=None,
                        help='Custom base URL for OpenAI‑compatible APIs (e.g., DeepSeek)')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--request_delay', type=float, default=0.1)
    parser.add_argument('--max_items', type=int, default=None)
    args = parser.parse_args()

    meta_dict = load_meta(args.meta_file)

    try:
        with open(args.output_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    items_to_process = [(asin, desc) for asin, desc in meta_dict.items() if asin not in cache]
    if args.max_items is not None and args.max_items < len(items_to_process):
        items_to_process = items_to_process[:args.max_items]
    print(f"Augmenting {len(items_to_process)} items using {args.num_workers} parallel workers...")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_item, asin, desc, args.model, args.api_key, args.base_url): asin
            for asin, desc in items_to_process
        }
        with tqdm(total=len(futures), desc="Augmenting") as pbar:
            for future in as_completed(futures):
                asin, aug = future.result()
                cache[asin] = aug
                if len(cache) % 200 == 0:
                    with open(args.output_cache, 'w', encoding='utf-8') as f:
                        json.dump(cache, f)
                pbar.update(1)
                time.sleep(args.request_delay)

    with open(args.output_cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f"Augmented descriptions saved to {args.output_cache}")

if __name__ == '__main__':
    main()
