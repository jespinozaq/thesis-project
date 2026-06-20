#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

# Output files (timestamped)
COLD_FILE="cold_start_low_exposure_$(date +%Y%m%d_%H%M%S).txt"
PROV_FILE="provider_fairness_$(date +%Y%m%d_%H%M%S).txt"

# List of models: tag, checkpoint, model_name
# Add your best‑tuned model here later (e.g., checkpoints_blair_base_fullaug_best_tuned/best_model.pt)
MODELS=(
  "zero_shot,blair-roberta-base-local,roberta-base"
  "base_500aug,checkpoints_blair_base_500aug/best_model.pt,roberta-base"
  "base_fullaug,checkpoints_blair_base_fullaug/best_model.pt,roberta-base"
  "base_deepseek_fullaug,checkpoints_blair_base_deepseek_fullaug/best_model.pt,roberta-base"
  "base_gpt4o_fullaug,checkpoints_blair_base_gpt4o_fullaug/best_model.pt,roberta-base"
  "large_fullaug,checkpoints_blair_large_fullaug/best_model.pt,blair-roberta-large-local"
  "llm_fullaug,checkpoints_llm_fullaug/best_model.pt,meta-llama/Llama-3.1-8B-Instruct"
  "best_tuned,checkpoints_blair_base_fullaug_best_tuned/best_model.pt,roberta-base"
)

echo "====== Low‑Exposure Cold‑Start (threshold=5) ======" | tee "$COLD_FILE"
for entry in "${MODELS[@]}"; do
    IFS=',' read -r tag cp mn <<< "$entry"
    echo "  ---- $tag ----" | tee -a "$COLD_FILE"
    if [ -f "$cp" ] || [ -d "$cp" ]; then
        python evaluate_cold_start.py \
            --model_path "$cp" \
            --model_name "$mn" \
            --test_seq processed_All_Beauty/test_seqs.json \
            --meta_file meta_All_Beauty.jsonl \
            --item_pop processed_All_Beauty/item_popularity.json \
            --topk 10 \
            --cold_threshold 5 | tee -a "$COLD_FILE"
    else
        echo "  Checkpoint not found: $cp" | tee -a "$COLD_FILE"
    fi
    echo "" | tee -a "$COLD_FILE"
done

echo "" | tee "$PROV_FILE"
echo "====== Provider‑Side Exposure Fairness ======" | tee -a "$PROV_FILE"
for entry in "${MODELS[@]}"; do
    IFS=',' read -r tag cp mn <<< "$entry"
    echo "  ---- $tag ----" | tee -a "$PROV_FILE"
    if [ -f "$cp" ] || [ -d "$cp" ]; then
        python provider_fairness_eval.py \
            --model_path "$cp" \
            --model_name "$mn" \
            --test_seq processed_All_Beauty/test_seqs.json \
            --meta_file meta_All_Beauty.jsonl \
            --topk 10 | tee -a "$PROV_FILE"
    else
        echo "  Checkpoint not found: $cp" | tee -a "$PROV_FILE"
    fi
    echo "" | tee -a "$PROV_FILE"
done

echo "Results saved to: $COLD_FILE and $PROV_FILE"
