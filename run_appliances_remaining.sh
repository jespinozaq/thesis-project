#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

echo "=== Experiment 4: LLM baselines ==="

# ChatGPT fair – max Tier‑5 throughput
echo "--- ChatGPT fair ---"
python evaluate_llm_baselines.py \
  --test_seqs processed_Amazon_Appliances/test_seqs.json \
  --meta_file meta_Appliances_filtered.jsonl \
  --item_pop processed_Amazon_Appliances/item_popularity.json \
  --backend openai \
  --model gpt-3.5-turbo \
  --prompt_mode fair \
  --topk 10 \
  --blair_model_path blair-roberta-base-local \
  --api_key $OPENAI_API_KEY \
  --num_workers 290 --request_delay 0.01 \
  > appliances_results/chatgpt_fair.txt

# DeepSeek fair – safe dynamic limit
echo "--- DeepSeek fair ---"
python evaluate_llm_baselines.py \
  --test_seqs processed_Amazon_Appliances/test_seqs.json \
  --meta_file meta_Appliances_filtered.jsonl \
  --item_pop processed_Amazon_Appliances/item_popularity.json \
  --backend openai \
  --model deepseek-chat \
  --prompt_mode fair \
  --topk 10 \
  --blair_model_path blair-roberta-base-local \
  --base_url https://api.deepseek.com \
  --api_key $DEEPSEEK_API_KEY \
  --num_workers 30 --request_delay 0.1 \
  > appliances_results/deepseek_fair.txt

echo "=== Experiment 5: Swap Combined ==="
python rerank_swap.py \
  --model_path checkpoints_blair_base_Appliances_fullaug/best_model.pt \
  --model_name roberta-base \
  --test_seq processed_Amazon_Appliances/test_seqs.json \
  --meta_file meta_Appliances_filtered.jsonl \
  --item_pop processed_Amazon_Appliances/item_popularity.json \
  --backend openai \
  --llm_model gpt-3.5-turbo \
  --api_key $OPENAI_API_KEY \
  --topk 10 --keep 8 --candidate_pool 30 \
  > appliances_results/swap_llm.txt

echo "=== Qualitative sampler ==="
python qualitative_sampler.py --meta_file meta_Appliances_filtered.jsonl \
  > appliances_results/qualitative.txt

echo "=== All remaining experiments finished ==="
