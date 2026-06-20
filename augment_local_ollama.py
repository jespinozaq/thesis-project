import json, argparse, time, requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

def build_prompt_A(original_desc):
    return (
        f"The description of an item is as follows: '{original_desc}'.\n"
        "To ensure this recommendation is fair and appealing to a diverse audience, generate an augmented "
        "description that:\n"
        "1. Highlights features of the item that are universally appealing, avoiding stereotypes.\n"
        "2. Describes how this item would be suitable for a senior user and a young adult.\n"
        "3. If there are any potential cultural, gender, or age‑related biases in the original description, "
        "rewrite a new, inclusive description for this item.\n"
        "Return only the augmented description, without any additional text."
    )

def process_item(asin, desc, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt_A(desc)}],
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False
    }
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post("http://localhost:8080/v1/chat/completions",
                              json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"].strip()
            return asin, content
        except Exception as e:
            print(f"Attempt {attempt} for {asin}: {e}")
            # Print the server's response body for debugging
            try:
                if 'r' in locals():
                    print("Server response body:", r.text[:500])
            except:
                pass
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
    parser.add_argument('--model', default='llama3.2:3b')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--request_delay', type=float, default=0)
    args = parser.parse_args()

    meta_dict = load_meta(args.meta_file)
    try:
        with open(args.output_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    items_to_process = [(asin, desc) for asin, desc in meta_dict.items() if asin not in cache]
    print(f"Augmenting {len(items_to_process)} items with {args.num_workers} workers...")

    write_lock = Lock()
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_item, asin, desc, args.model): asin
                   for asin, desc in items_to_process}
        with tqdm(total=len(futures), desc="Augmenting") as pbar:
            for future in as_completed(futures):
                asin, aug = future.result()
                with write_lock:
                    cache[asin] = aug
                    if len(cache) % 200 == 0:
                        with open(args.output_cache, 'w', encoding='utf-8') as f:
                            json.dump(cache, f)
                pbar.update(1)
                time.sleep(args.request_delay)

    with open(args.output_cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f"Done. Cache saved to {args.output_cache}")

if __name__ == '__main__':
    main()
