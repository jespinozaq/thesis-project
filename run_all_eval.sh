#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

echo "===== MASTER EVALUATION (Config A) ====="
cp eval_config_A.json eval_config.json
python run_all_evaluations.py
mv evaluation_results.csv results_A.csv
echo "results_A.csv saved."

echo "===== MASTER EVALUATION (Config B) ====="
cp eval_config_B.json eval_config.json
python run_all_evaluations.py
mv evaluation_results.csv results_B.csv
echo "results_B.csv saved."

echo "===== STRATIFIED RECALL ====="
for tag in base_500aug base_fullaug base_deepseek_fullaug base_gpt4o_fullaug large_fullaug llm_fullaug; do
    case $tag in
        base_500aug)          CP="checkpoints_blair_base_500aug/best_model.pt" ; MN="roberta-base" ;;
        base_fullaug)         CP="checkpoints_blair_base_fullaug/best_model.pt" ; MN="roberta-base" ;;
        base_deepseek_fullaug) CP="checkpoints_blair_base_deepseek_fullaug/best_model.pt" ; MN="roberta-base" ;;
        base_gpt4o_fullaug)   CP="checkpoints_blair_base_gpt4o_fullaug/best_model.pt" ; MN="roberta-base" ;;
        large_fullaug)        CP="checkpoints_blair_large_fullaug/best_model.pt" ; MN="roberta-base" ;;
        llm_fullaug)          CP="checkpoints_llm_fullaug/best_model.pt" ; MN="meta-llama/Llama-3.1-8B-Instruct" ;;
    esac
    if [ -f "$CP" ]; then
        python stratified_recall.py --model_path "$CP" --model_name "$MN" \
            --test_seq processed_All_Beauty/test_seqs.json --meta_file meta_All_Beauty.jsonl \
            --item_pop processed_All_Beauty/item_popularity.json --topk 10 --output_tag "$tag"
    else
        echo "Skipping $tag (checkpoint not found)"
    fi
done
echo "Stratified recall complete."

echo "===== QUALITATIVE EXAMPLES ====="
python qualitative_sampler.py > qualitative_examples.txt
echo "Qualitative examples saved."
