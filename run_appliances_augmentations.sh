#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

echo "===== GPT-4o-mini full cache (Appliances) ====="
python augment_descriptions_parallel.py \
  --meta_file meta_Appliances_filtered.jsonl \
  --output_cache augmented_descriptions_Appliances.json \
  --model gpt-4o-mini \
  --api_key $OPENAI_API_KEY \
  --num_workers 30 --request_delay 0.05

echo "===== Appliances augmentation done ====="
