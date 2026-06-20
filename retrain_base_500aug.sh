#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

echo "Retraining BLAIR‑base + 500‑aug (gpt‑4o‑mini) …"
python train_framework.py \
  --train_pairs_file processed_All_Beauty/train_pairs.jsonl \
  --meta_file meta_All_Beauty.jsonl \
  --item_pop_file processed_All_Beauty/item_popularity.json \
  --use_augmentation \
  --augmentation_cache augmented_descriptions_500.json \
  --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 \
  --output_dir checkpoints_blair_base_500aug \
  --model_name blair-roberta-base-local \
  --batch_size 256 --gradient_accumulation_steps 1 \
  --epochs 20 --cpt_epochs 1 --lr 5e-5 \
  --temperature 0.05 --beta 0.2 --topk 10 \
  --fp16 --val_seq processed_All_Beauty/val_seqs.json \
  --max_seq_length 128 --seed 42

echo "Training complete. Running stratified recall for this model …"
python stratified_recall.py \
  --model_path checkpoints_blair_base_500aug/best_model.pt \
  --model_name roberta-base \
  --test_seq processed_All_Beauty/test_seqs.json \
  --meta_file meta_All_Beauty.jsonl \
  --item_pop processed_All_Beauty/item_popularity.json \
  --topk 10 \
  --output_tag base_500aug

echo "Done. The stratified recall file recommendations_base_500aug.json has been created."
echo "You may now re‑run run_all_eval.sh to obtain an updated evaluation_results.csv."
