#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project
export HF_TOKEN="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"

echo "START: BLAIR‑base + DeepSeek full‑aug"
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_deepseek.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_deepseek_fullaug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 3 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42
echo "DONE: DeepSeek"

echo "START: BLAIR‑base + GPT‑4o full‑aug"
python train_framework.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_gpt4o.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_base_gpt4o_fullaug --model_name blair-roberta-base-local --batch_size 256 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 3 --lr 5e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42
echo "DONE: GPT‑4o"

echo "START: BLAIR‑large + full‑aug (GPT‑4o‑mini)"
python train_large.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_full.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_blair_large_fullaug --model_name blair-roberta-large-local --batch_size 32 --gradient_accumulation_steps 1 --epochs 20 --cpt_epochs 3 --lr 2e-5 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42
echo "DONE: Large"

echo "START: LLM‑as‑embedder (Llama‑3.1‑8B) + full‑aug"
python train_llm.py --train_pairs_file processed_All_Beauty/train_pairs.jsonl --meta_file meta_All_Beauty.jsonl --item_pop_file processed_All_Beauty/item_popularity.json --use_augmentation --augmentation_cache augmented_descriptions_full.json --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 --output_dir checkpoints_llm_fullaug --model_name meta-llama/Llama-3.1-8B-Instruct --batch_size 4 --gradient_accumulation_steps 8 --epochs 20 --cpt_epochs 3 --lr 2e-4 --temperature 0.05 --beta 0.2 --topk 10 --fp16 --val_seq processed_All_Beauty/val_seqs.json --max_seq_length 128 --seed 42
echo "DONE: LLM‑as‑Embedder"
echo "ALL TRAININGS COMPLETE"
