#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

# ---------- user‑configurable ----------
GPU_SAFE_MODE=true   # set to false when LLM training is finished
RAW_META="raw_data/meta_Appliances.jsonl"
FILTERED_META="meta_Appliances_filtered.jsonl"
AUG_CACHE="augmented_descriptions_Appliances.json"
TRAIN_PAIRS="processed_Amazon_Appliances/train_pairs.jsonl"
ITEM_POP="processed_Amazon_Appliances/item_popularity.json"
VAL_SEQ="processed_Amazon_Appliances/val_seqs.json"
TEST_SEQ="processed_Amazon_Appliances/test_seqs.json"
OUTPUT_DIR="checkpoints_blair_base_Appliances_fullaug"
RESULTS_DIR="appliances_results"
BEST_MODEL="${OUTPUT_DIR}/best_model.pt"
RESULTS_CSV="${RESULTS_DIR}/appliances_all_results.csv"

# ---------- helpers ----------
check_space() {
    local required_kb=$1
    local path=$2
    local avail_kb=$(df --output=avail "$path" | tail -1)
    if [ "$avail_kb" -lt "$required_kb" ]; then
        echo "ERROR: Not enough space in $path (need ${required_kb} KB, have ${avail_kb} KB)."
        exit 1
    fi
}

gpu_available() {
    if nvidia-smi --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | awk '{sum+=$1} END{exit sum>1024}'; then
        return 0
    else
        return 1
    fi
}

# ---------- Pre‑processing (safe now) ----------
echo "=== Step 1: Filter meta ==="
if [ -f "$FILTERED_META" ]; then
    echo "$FILTERED_META exists, skipping."
else
    check_space 500000 "/workspace"
    python filter_meta.py \
        --train_pairs "$TRAIN_PAIRS" \
        --meta_in "$RAW_META" \
        --meta_out "$FILTERED_META"
fi

echo "=== Step 2: Augment descriptions ==="
if [ -f "$AUG_CACHE" ]; then
    echo "$AUG_CACHE exists, skipping."
else
    check_space 2000000 "/workspace"
    python augment_descriptions_parallel.py \
        --meta_file "$FILTERED_META" \
        --output_cache "$AUG_CACHE" \
        --model gpt-4o-mini \
        --api_key "$OPENAI_API_KEY" \
        --num_workers 30 \
        --request_delay 0.05
fi

# ---------- GPU step check ----------
if [ "$GPU_SAFE_MODE" = "true" ]; then
    if ! gpu_available; then
        echo ""
        echo "⚠️  GPU busy — skipping training and evaluations. Re‑run with GPU_SAFE_MODE=false when GPU is free."
        exit 0
    fi
fi

mkdir -p "$RESULTS_DIR"

# ---------- Experiment 1: Fine‑tuned BLAIR‑base ----------
echo "=== Experiment 1: Fine‑tuned BLAIR‑base (full aug) ==="
if [ -f "$BEST_MODEL" ]; then
    echo "$BEST_MODEL exists, skipping training."
else
    check_space 5000000 "/workspace"
    python train_framework.py \
        --train_pairs_file "$TRAIN_PAIRS" \
        --meta_file "$FILTERED_META" \
        --item_pop_file "$ITEM_POP" \
        --use_augmentation \
        --augmentation_cache "$AUG_CACHE" \
        --lambda_aug 0.1 --lambda_pop 0.1 --gamma_div 0.05 \
        --output_dir "$OUTPUT_DIR" \
        --model_name blair-roberta-base-local \
        --batch_size 256 --gradient_accumulation_steps 1 \
        --epochs 20 --cpt_epochs 2 --lr 5e-5 \
        --temperature 0.05 --beta 0.2 --topk 10 \
        --fp16 --val_seq "$VAL_SEQ" \
        --max_seq_length 128 --seed 42
fi

# ---------- Experiment 2: Zero‑shot BLAIR ----------
echo "=== Experiment 2: Zero‑shot BLAIR ==="
ZERO_RESULT="${RESULTS_DIR}/zeroshot_blair.csv"
if [ -f "$ZERO_RESULT" ]; then
    echo "$ZERO_RESULT exists, skipping."
else
    python evaluate_sequential.py \
        --model_path blair-roberta-base-local \
        --model_name roberta-base \
        --test_seq "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --topk 10 > "${RESULTS_DIR}/zeroshot_blair_metrics.txt"
    python -c "
import csv, re
with open('${RESULTS_DIR}/zeroshot_blair_metrics.txt') as f: text = f.read()
hr   = re.search(r'HR@10:\s+([0-9.]+)', text).group(1) if re.search(r'HR@10:', text) else ''
ndcg = re.search(r'NDCG@10:\s+([0-9.]+)', text).group(1) if re.search(r'NDCG@10:', text) else ''
pop  = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text).group(1) if re.search(r'Average Log Popularity Difference:', text) else ''
div  = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', text).group(1) if re.search(r'Intra-list Diversity', text) else ''
cov  = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text).group(1) if re.search(r'Catalogue Coverage:', text) else ''
with open('${ZERO_RESULT}','w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Experiment','HR@10','NDCG@10','LogPopDiff','Diversity','Coverage'])
    writer.writeheader()
    writer.writerow({'Experiment':'Zero‑shot BLAIR (Appliances)','HR@10':hr,'NDCG@10':ndcg,'LogPopDiff':pop,'Diversity':div,'Coverage':cov})
"
fi

# ---------- Experiment 3: Fine‑tuned BLAIR‑base evaluation ----------
echo "=== Experiment 3: Evaluate fine‑tuned base + stratified recall ==="
if [ -f "${RESULTS_DIR}/base_metrics.txt" ]; then
    echo "Base metrics already computed, skipping."
else
    python evaluate_finetuned.py \
        --model_path "$BEST_MODEL" \
        --model_name roberta-base \
        --test_seq "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --topk 10 > "${RESULTS_DIR}/base_metrics.txt"
    python -c "
import csv, re
with open('${RESULTS_DIR}/base_metrics.txt') as f: text = f.read()
hr   = re.search(r'HR@10:\s+([0-9.]+)', text).group(1) if re.search(r'HR@10:', text) else ''
ndcg = re.search(r'NDCG@10:\s+([0-9.]+)', text).group(1) if re.search(r'NDCG@10:', text) else ''
pop  = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text).group(1) if re.search(r'Average Log Popularity Difference:', text) else ''
div  = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', text).group(1) if re.search(r'Intra-list Diversity', text) else ''
cov  = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text).group(1) if re.search(r'Catalogue Coverage:', text) else ''
row = {'Experiment':'BLAIR Base (Appliances, full aug)','HR@10':hr,'NDCG@10':ndcg,'LogPopDiff':pop,'Diversity':div,'Coverage':cov}
with open('${RESULTS_CSV}','a',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if f.tell()==0: writer.writeheader()
    writer.writerow(row)
"
fi

# ---------- Stratified recall for base model ----------
echo "=== Stratified recall for base model ==="
if [ -f "recommendations_appliances_base_fullaug.json" ]; then
    echo "Stratified recall already done, skipping."
else
    python stratified_recall.py \
        --model_path "$BEST_MODEL" \
        --model_name roberta-base \
        --test_seq "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --topk 10 --output_tag appliances_base_fullaug
fi

# ---------- Experiment 4: LLM Baselines (gentle fair prompt) ----------
echo "=== Experiment 4: LLM baselines (ChatGPT fair, DeepSeek fair) ==="
LLM_RESULT="${RESULTS_DIR}/llm_baselines.csv"
if [ -f "$LLM_RESULT" ]; then
    echo "$LLM_RESULT exists, skipping."
else
    # ChatGPT fair
    python evaluate_llm_baselines.py \
        --test_seqs "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --backend openai \
        --model gpt-3.5-turbo \
        --prompt_mode fair \
        --topk 10 \
        --blair_model_path blair-roberta-base-local \
        --api_key "$OPENAI_API_KEY" > "${RESULTS_DIR}/chatgpt_fair.txt"

    # DeepSeek fair
    python evaluate_llm_baselines.py \
        --test_seqs "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --backend openai \
        --model deepseek-chat \
        --prompt_mode fair \
        --topk 10 \
        --blair_model_path blair-roberta-base-local \
        --base_url https://api.deepseek.com \
        --api_key "$DEEPSEEK_API_KEY" > "${RESULTS_DIR}/deepseek_fair.txt"

    # Extract and save
    python -c "
import csv, re
def extract(filename):
    with open(filename) as f: text = f.read()
    hr   = re.search(r'HR@10:\s+([0-9.]+)', text).group(1) if re.search(r'HR@10:', text) else ''
    ndcg = re.search(r'NDCG@10:\s+([0-9.]+)', text).group(1) if re.search(r'NDCG@10:', text) else ''
    pop  = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text).group(1) if re.search(r'Average Log Popularity Difference:', text) else ''
    div  = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', text).group(1) if re.search(r'Intra-list Diversity', text) else ''
    cov  = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text).group(1) if re.search(r'Catalogue Coverage:', text) else ''
    return hr, ndcg, pop, div, cov

rows = []
for name, fname in [('ChatGPT fair (Appliances)', '${RESULTS_DIR}/chatgpt_fair.txt'),
                    ('DeepSeek fair (Appliances)', '${RESULTS_DIR}/deepseek_fair.txt')]:
    hr, ndcg, pop, div, cov = extract(fname)
    rows.append({'Experiment': name, 'HR@10': hr, 'NDCG@10': ndcg, 'LogPopDiff': pop, 'Diversity': div, 'Coverage': cov})

with open('${LLM_RESULT}','w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
"
fi

# ---------- Experiment 5: Swap Combined ----------
echo "=== Experiment 5: Swap Combined (BLAIR + ChatGPT fair) ==="
SWAP_RESULT="${RESULTS_DIR}/swap_combined.csv"
if [ -f "$SWAP_RESULT" ]; then
    echo "$SWAP_RESULT exists, skipping."
else
    # Baseline
    python rerank_swap.py \
        --model_path "$BEST_MODEL" \
        --model_name roberta-base \
        --test_seq "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --backend openai \
        --llm_model gpt-3.5-turbo \
        --api_key "$OPENAI_API_KEY" \
        --topk 10 --keep 8 --candidate_pool 30 \
        --baseline_only > "${RESULTS_DIR}/swap_baseline.txt"

    # Swap
    python rerank_swap.py \
        --model_path "$BEST_MODEL" \
        --model_name roberta-base \
        --test_seq "$TEST_SEQ" \
        --meta_file "$FILTERED_META" \
        --item_pop "$ITEM_POP" \
        --backend openai \
        --llm_model gpt-3.5-turbo \
        --api_key "$OPENAI_API_KEY" \
        --topk 10 --keep 8 --candidate_pool 30 > "${RESULTS_DIR}/swap_llm.txt"

    python -c "
import csv, re
def extract(filename):
    with open(filename) as f: text = f.read()
    hr   = re.search(r'HR@10:\s+([0-9.]+)', text).group(1) if re.search(r'HR@10:', text) else ''
    ndcg = re.search(r'NDCG@10:\s+([0-9.]+)', text).group(1) if re.search(r'NDCG@10:', text) else ''
    pop  = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text).group(1) if re.search(r'Average Log Popularity Difference:', text) else ''
    div  = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', text).group(1) if re.search(r'Intra-list Diversity', text) else ''
    cov  = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text).group(1) if re.search(r'Catalogue Coverage:', text) else ''
    return hr, ndcg, pop, div, cov

hr_b, ndcg_b, pop_b, div_b, cov_b = extract('${RESULTS_DIR}/swap_baseline.txt')
hr_s, ndcg_s, pop_s, div_s, cov_s = extract('${RESULTS_DIR}/swap_llm.txt')

rows = [
    {'Experiment':'Swap Baseline (pool=30) Appliances','HR@10':hr_b,'NDCG@10':ndcg_b,'LogPopDiff':pop_b,'Diversity':div_b,'Coverage':cov_b},
    {'Experiment':'Swap Combined (ChatGPT fair) Appliances','HR@10':hr_s,'NDCG@10':ndcg_s,'LogPopDiff':pop_s,'Diversity':div_s,'Coverage':cov_s}
]
with open('${SWAP_RESULT}','w',newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
"
fi

# ---------- Qualitative sampler ----------
echo "=== Qualitative sampler ==="
if [ -f "${RESULTS_DIR}/qualitative.txt" ]; then
    echo "Qualitative examples already generated, skipping."
else
    python qualitative_sampler.py > "${RESULTS_DIR}/qualitative.txt"
fi

echo ""
echo "========== Appliances full pipeline finished =========="
echo "Results are in ${RESULTS_CSV}, ${LLM_RESULT}, ${SWAP_RESULT}, ${ZERO_RESULT}, and ${RESULTS_DIR}/qualitative.txt"
