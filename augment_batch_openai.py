import json, time, argparse, os
from openai import OpenAI

# ---------- Configuration ----------
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

# ---------- Batch helpers ----------
def create_batch_input_file(items, model, file_prefix="batch_input"):
    """Write a .jsonl file with batch requests. Returns (file_path, request_ids)."""
    requests = []
    for asin, desc in items:
        request = {
            "custom_id": asin,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": build_prompt_A(desc)}],
                "temperature": 0.0,
                "max_tokens": 200
            }
        }
        requests.append(request)

    file_path = f"{file_prefix}_{int(time.time())}.jsonl"
    with open(file_path, 'w', encoding='utf-8') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    print(f"Created batch input file with {len(requests)} requests: {file_path}")
    return file_path

def upload_and_submit_batch(client, input_file_path):
    """Upload file, create batch, return batch object."""
    # Upload file
    with open(input_file_path, 'rb') as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    print(f"Uploaded file ID: {file_obj.id}")

    # Create batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch created: {batch.id}, status: {batch.status}")
    return batch

def wait_for_batch(client, batch_id, check_interval=30):
    """Poll until batch completes, then return the batch object."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"Batch {batch_id}: {status}")
        if status in ("completed", "failed", "expired", "cancelled"):
            return batch
        time.sleep(check_interval)

def download_and_update_cache(client, batch, cache, output_cache_path):
    """Download output, parse augmented descriptions, update cache, save."""
    if batch.status != "completed":
        print(f"Batch not completed (status: {batch.status}); cannot retrieve results.")
        return

    # Download output file
    output_file_id = batch.output_file_id
    if not output_file_id:
        print("No output file – batch may have failed.")
        return

    content = client.files.content(output_file_id)
    lines = content.text.strip().split('\n')
    print(f"Processing {len(lines)} output lines...")

    updated = 0
    for line in lines:
        try:
            result = json.loads(line)
            custom_id = result['custom_id']   # ASIN
            response_body = result['response']['body']
            if 'choices' in response_body and len(response_body['choices']) > 0:
                aug = response_body['choices'][0]['message']['content'].strip()
                cache[custom_id] = aug
                updated += 1
            else:
                # Fallback to original description (store original description later)
                # But original desc not in cache now... skip for now, or just mark as error
                print(f"Warning: No valid response for {custom_id}, skipping.")
                # Optionally keep original (get it from meta later hopefully)
        except Exception as e:
            print(f"Error processing line for custom_id {line.get('custom_id','?')}: {e}")
            continue

    # Save cache
    with open(output_cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    print(f"Cache updated with {updated} new entries. Saved to {output_cache_path}")

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--output_cache', required=True)
    parser.add_argument('--model', default='gpt-4o-mini')
    parser.add_argument('--api_key', required=True)
    args = parser.parse_args()

    client = OpenAI(api_key=args.api_key)

    # Load meta and existing cache
    meta_dict = load_meta(args.meta_file)
    try:
        with open(args.output_cache, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    except FileNotFoundError:
        cache = {}

    # Items yet to process
    to_process = [(asin, meta_dict[asin]) for asin in meta_dict if asin not in cache]
    print(f"Total items to augment: {len(to_process)}")

    if not to_process:
        print("All items already augmented. Exiting.")
        return

    # Split into chunks of 50k (OpenAI batch limit)
    chunk_size = 50000
    for i in range(0, len(to_process), chunk_size):
        chunk = to_process[i:i+chunk_size]
        print(f"\n--- Processing chunk {i//chunk_size + 1} with {len(chunk)} items ---")
        file_path = create_batch_input_file(chunk, args.model)
        batch = upload_and_submit_batch(client, file_path)
        print("Waiting for batch to complete (this may take a while)...")
        batch = wait_for_batch(client, batch.id)
        download_and_update_cache(client, batch, cache, args.output_cache)
        # Clean up local file
        os.remove(file_path)
        # Short pause before next batch
        time.sleep(5)

    print("All batches done. Final cache saved.")

if __name__ == '__main__':
    main()
