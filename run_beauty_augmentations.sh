#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

echo "===== GPT-4o-mini 500-item cache ====="
python augment_descriptions_parallel.py \
  --meta_file meta_All_Beauty.jsonl \
  --output_cache augmented_descriptions_500.json \
  --model gpt-4o-mini \
  --api_key $OPENAI_API_KEY \
  --num_workers 20 --max_items 500 \
  --request_delay 0.1

echo "===== GPT-4o-mini full cache ====="
python augment_descriptions_parallel.py \
  --meta_file meta_All_Beauty.jsonl \
  --output_cache augmented_descriptions_full.json \
  --model gpt-4o-mini \
  --api_key $OPENAI_API_KEY \
  --num_workers 30 --request_delay 0.05

echo "===== GPT-4o full cache ====="
python augment_descriptions_parallel.py \
  --meta_file meta_All_Beauty.jsonl \
  --output_cache augmented_descriptions_gpt4o.json \
  --model gpt-4o \
  --api_key $OPENAI_API_KEY \
  --num_workers 30 --request_delay 0.1

echo "===== DeepSeek-chat full cache ====="
python augment_descriptions_parallel.py \
  --meta_file meta_All_Beauty.jsonl \
  --output_cache augmented_descriptions_deepseek.json \
  --model deepseek-chat \
  --api_key $DEEPSEEK_API_KEY \
  --base_url https://api.deepseek.com \
  --num_workers 20 --request_delay 0.1

echo "===== All Beauty augmentations done ====="
