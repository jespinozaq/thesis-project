#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project
export HF_TOKEN="hf_..."   # if needed for LLM model
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# -------------------------------------------------------------------
# 1. Train all fine-tuned models (sequential, each uses full GPU)
# -------------------------------------------------------------------
echo "===== BLAIR-base 500-aug ====="
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_500.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_500aug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 1 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

echo "===== BLAIR-base full-aug (GPT-4o-mini) ====="
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_full.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_fullaug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 2 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

echo "===== BLAIR-base DeepSeek full-aug ====="
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_deepseek.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_deepseek_fullaug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 2 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

echo "===== BLAIR-base GPT-4o full-aug ====="
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_gpt4o.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_gpt4o_fullaug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 2 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

echo "===== BLAIR-large full-aug ====="
python train_large.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_full.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_large_fullaug --model_name blair-roberta-large-local --batch_size 32 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 2 --lr 2e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

echo "===== LLM-as-Embedder (Llama-3.1-8B) full-aug ====="
export HF_TOKEN="hf_..."
python train_llm.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_full.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_llm_fullaug --model_name meta-llama/Llama-3.1-8B-Instruct --batch_size 4 --gradient_accumulation_steps 8 --epochs 20 --cpt_epochs 2 --lr 2e-4 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42

# -------------------------------------------------------------------
# 2. Master evaluations (Config A & B)
# -------------------------------------------------------------------
echo "===== Config A evaluation ====="
cp eval_config_A.json eval_config.json
python run_all_evaluations.py
mv evaluation_results.csv results_A.csv

echo "===== Config B evaluation ====="
cp eval_config_B.json eval_config.json
python run_all_evaluations.py
mv evaluation_results.csv results_B.csv

# -------------------------------------------------------------------
# 3. Multi-seed evaluation (non-LLM)
# -------------------------------------------------------------------
echo "===== Multi-seed evaluation ====="
./run_multiseed_eval.sh

# -------------------------------------------------------------------
# 4. Loss-comparison experiment
# -------------------------------------------------------------------
echo "===== Loss comparison ====="
python compare_losses.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --val_seq processed_All_Beauty/val_seqs.json --output_dir checkpoints_loss_comparison --model_name blair-roberta-base-local --batch_size 256 --epochs 20 --lr 5e-5 --max_seq_length 128 --fp16 --losses infonce bpr debiased bc sce ipw_sce --run_full_eval --eval_script evaluate_finetuned.py --test_seq processed_All_Beauty/test_seqs.json --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --topk 10

# -------------------------------------------------------------------
# 5. Stratified recall & qualitative sampler
# -------------------------------------------------------------------
echo "===== Stratified recall ====="
for tag in base_500aug base_fullaug base_deepseek_fullaug base_gpt4o_fullaug large_fullaug llm_fullaug; do
    case $tag in
        base_500aug)          CP="checkpoints_blair_base_500aug/best_model.pt"; MN="roberta-base" ;;
        base_fullaug)         CP="checkpoints_blair_base_fullaug/best_model.pt"; MN="roberta-base" ;;
        base_deepseek_fullaug) CP="checkpoints_blair_base_deepseek_fullaug/best_model.pt"; MN="roberta-base" ;;
        base_gpt4o_fullaug)   CP="checkpoints_blair_base_gpt4o_fullaug/best_model.pt"; MN="roberta-base" ;;
        large_fullaug)        CP="checkpoints_blair_large_fullaug/best_model.pt"; MN="roberta-base" ;;
        llm_fullaug)          CP="checkpoints_llm_fullaug/best_model.pt"; MN="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac
    if [ -f "$CP" ]; then
        python stratified_recall.py --model_path "$CP" --model_name "$MN" \
            --test_seq processed_All_Beauty/test_seqs.json --meta_file meta_All_Beauty.jsonl \
            --item_pop processed_All_Beauty/item_popularity.json --topk 10 --output_tag "$tag"
    fi
done

echo "===== Qualitative sampler ====="
python qualitative_sampler.py > qualitative_examples.txt

echo "===== Beauty full pipeline finished ====="
