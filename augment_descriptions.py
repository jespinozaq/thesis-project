import json
import argparse
from tqdm import tqdm

# ---------- LLM backends ----------
def get_openai_response(prompt, model, api_key):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()

def get_anthropic_response(prompt, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=200,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text.strip()

def get_ollama_response(prompt, model, url="http://localhost:11434/api/chat"):
    import requests
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False}
    r = requests.post(url, json=payload)
    data = r.json()
    # Older Ollama versions nest under 'message', newer may use 'choices'
    if 'message' in data:
        return data['message']['content'].strip()
    elif 'choices' in data and len(data['choices']) > 0:
        return data['choices'][0]['message']['content'].strip()
    else:
        raise KeyError(f"Unexpected response: {data}")
# ------------------------------------

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
    parser.add_argument('--meta_file', required=True, help='Path to meta JSONL file')
    parser.add_argument('--output_cache', required=True, help='Path to save augmented descriptions JSON')
    parser.add_argument('--backend', required=True, choices=['openai', 'anthropic', 'ollama'],
                        help='LLM backend to use')
    parser.add_argument('--model', default='gpt-3.5-turbo', help='Model name')
    parser.add_argument('--api_key', default=None, help='API key for OpenAI/Anthropic')
    parser.add_argument('--ollama_url', default='http://localhost:11434/api/chat', help='Ollama chat endpoint')
    args = parser.parse_args()

    meta_dict = load_meta(args.meta_file)

    # Load existing cache if any
    try:
        with open(args.output_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    asins = [a for a in meta_dict if a not in cache]
    print(f"Augmenting {len(asins)} items...")

    for asin in tqdm(asins):
        desc = meta_dict[asin]
        prompt = build_prompt_A(desc)
        try:
            if args.backend == 'openai':
                aug = get_openai_response(prompt, args.model, args.api_key)
            elif args.backend == 'anthropic':
                aug = get_anthropic_response(prompt, args.model, args.api_key)
            else:
                aug = get_ollama_response(prompt, args.model, args.ollama_url)
            cache[asin] = aug
        except Exception as e:
            print(f"Error for {asin}: {e}")
            cache[asin] = desc  # fallback

        # Save incrementally every 50 items
        if len(cache) % 50 == 0:
            with open(args.output_cache, 'w', encoding='utf-8') as f:
                json.dump(cache, f)

    with open(args.output_cache, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f"Augmented descriptions saved to {args.output_cache}")

if __name__ == '__main__':
    main()