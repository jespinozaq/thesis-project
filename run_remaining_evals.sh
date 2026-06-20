#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

export HF_TOKEN="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"

# -------------------------------------------------------------------
# 1. Multi‑seed evaluation (non‑LLM experiments only)
# -------------------------------------------------------------------
echo "===== Multi‑seed evaluation ====="
./run_multiseed_eval.sh

# -------------------------------------------------------------------
# 2. Loss‑comparison experiment
# -------------------------------------------------------------------
echo "===== Loss comparison ====="
python compare_losses.py \
  --train_pairs_file processed_All_Beauty/train_pairs.jsonl \
  --meta_file meta_All_Beauty.jsonl \
  --item_pop_file processed_All_Beauty/item_popularity.json \
  --val_seq processed_All_Beauty/val_seqs.json \
  --output_dir checkpoints_loss_comparison \
  --model_name blair-roberta-base-local \
  --batch_size 256 --epochs 20 --lr 5e-5 --max_seq_length 128 --fp16 \
  --losses infonce bpr debiased bc sce ipw_sce \
  --run_full_eval \
  --eval_script evaluate_finetuned.py \
  --test_seq processed_All_Beauty/test_seqs.json \
  --meta_file meta_All_Beauty.jsonl \
  --item_pop_file processed_All_Beauty/item_popularity.json \
  --topk 10

# -------------------------------------------------------------------
# 3. Stratified recall (all checkpoints)
# -------------------------------------------------------------------
echo "===== Stratified recall ====="
for tag in base_500aug base_fullaug base_deepseek_fullaug base_gpt4o_fullaug large_fullaug llm_fullaug; do
    case $tag in
        base_500aug)          CP="checkpoints_blair_base_500aug/best_model.pt"; MN="roberta-base" ;;
        base_fullaug)         CP="checkpoints_blair_base_fullaug/best_model.pt"; MN="roberta-base" ;;
        base_deepseek_fullaug) CP="checkpoints_blair_base_deepseek_fullaug/best_model.pt"; MN="roberta-base" ;;
        base_gpt4o_fullaug)   CP="checkpoints_blair_base_gpt4o_fullaug/best_model.pt"; MN="roberta-base" ;;
        large_fullaug)        CP="checkpoints_blair_large_fullaug/best_model.pt"; MN="blair-roberta-large-local" ;;
        llm_fullaug)          CP="checkpoints_llm_fullaug/best_model.pt"; MN="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac
    if [ -f "$CP" ]; then
        python stratified_recall.py --model_path "$CP" --model_name "$MN" \
            --test_seq processed_All_Beauty/test_seqs.json --meta_file meta_All_Beauty.jsonl \
            --item_pop processed_All_Beauty/item_popularity.json --topk 10 --output_tag "$tag"
    fi
done

# -------------------------------------------------------------------
# 4. Qualitative sampler
# -------------------------------------------------------------------
echo "===== Qualitative sampler ====="
python qualitative_sampler.py > qualitative_examples.txt

echo "===== All remaining evaluations finished ====="
