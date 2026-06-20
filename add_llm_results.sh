#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

echo "Evaluating LLM‑as‑embedder …"
python evaluate_finetuned.py \
  --model_path checkpoints_llm_fullaug/best_model.pt \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --test_seq processed_All_Beauty/test_seqs.json \
  --meta_file meta_All_Beauty.jsonl \
  --item_pop processed_All_Beauty/item_popularity.json \
  --topk 10 > llm_results.txt

echo "Extracting metrics …"
python -c "
import csv, re

with open('llm_results.txt') as f:
    text = f.read()

hr   = re.search(r'HR@10:\s+([0-9.]+)', text)
hr   = hr.group(1) if hr else ''
ndcg = re.search(r'NDCG@10:\s+([0-9.]+)', text)
ndcg = ndcg.group(1) if ndcg else ''
pop  = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text)
pop  = pop.group(1) if pop else ''
div  = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', text)
div  = div.group(1) if div else ''
cov  = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text)
cov  = cov.group(1) if cov else ''

with open('results_B_partial.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['LLM‑as‑Embedder (Llama‑3.1‑8B, full aug)', hr, ndcg, pop, div, cov])
print('LLM row appended to results_B_partial.csv')
"
echo "Done. You can now rename results_B_partial.csv to results_B.csv if desired."
