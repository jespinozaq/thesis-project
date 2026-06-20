import openai
import json
from tqdm import tqdm

def augment_item_descriptions(meta_file, api_key, output_cache, model="gpt-3.5-turbo"):
    """
    Offline augmentation of item descriptions using Prompt A.
    """
    openai.api_key = api_key

    # Load existing cache if any
    try:
        with open(output_cache, 'r') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    # Load all items
    items = {}
    with open(meta_file, 'r') as f:
        for line in f:
            meta = json.loads(line.strip())
            asin = meta['parent_asin']
            if asin in cache:
                continue
            # Original description string (same as in dataset)
            meta_parts = [meta.get('title', '')]
            if 'features' in meta:
                meta_parts.extend(meta['features'])
            if 'description' in meta:
                meta_parts.extend(meta['description'])
            desc = ' '.join(meta_parts).replace('\n', ' ').strip()
            items[asin] = desc

    for asin, desc in tqdm(items.items(), desc="Augmenting item descriptions"):
        prompt = f"""The description of an item is as follows: '{desc}'. To ensure this recommendation is fair and inclusive, generate an augmented description that highlights universally appealing features and describes suitability for diverse groups (e.g., a senior user, a young adult). Do not change factual attributes."""
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            augmented = response['choices'][0]['message']['content'].strip()
            cache[asin] = augmented
        except Exception as e:
            print(f"Error for {asin}: {e}")
            cache[asin] = desc  # fallback

        # Save incrementally
        with open(output_cache, 'w') as f:
            json.dump(cache, f)

    print(f"Augmentation cache saved to {output_cache}")

def fair_rerank_with_llm(user_id, history_items, candidate_items, api_key, model="gpt-3.5-turbo"):
    """
    Use Prompt B to re-rank a shortlist of candidates for fairness.
    Returns the selected item ID.
    """
    openai.api_key = api_key
    history_str = ", ".join(history_items)
    candidates_str = "\n".join([f"{i+1}. {item['title']} - {item['description']}" for i, item in enumerate(candidate_items)])

    prompt = f"""We want to make a fair recommendation for user_{user_id} who has previously interacted with: {history_str}. From the following candidate items:
{candidates_str}
Select the best item that aligns with the user's interests while ensuring it does not reinforce stereotypes based on the user's history and represents a diverse set of characteristics. Output only the number of the selected item."""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=5
    )
    selection = response['choices'][0]['message']['content'].strip()
    try:
        idx = int(selection) - 1
        return candidate_items[idx]['id']
    except:
        return candidate_items[0]['id']  # fallback