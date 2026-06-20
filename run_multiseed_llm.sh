#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

SEEDS="42 43 44"
COMBINED_CSV="llm_multiseed_combined.csv"

# Collect per‑seed metrics
ALL_CSVS=""
for seed in $SEEDS; do
    echo "===== LLM multi‑seed: seed $seed ====="

    # Build a config containing only LLM + combined experiments from both configs
    python -c "
import json, copy

experiments = []
for cfg_file in ['eval_config_A.json', 'eval_config_B.json']:
    with open(cfg_file) as f:
        cfg = json.load(f)
    for exp in cfg['experiments']:
        if exp['type'] in ('llm', 'combined'):
            e = copy.deepcopy(exp)
            e['seed'] = $seed
            experiments.append(e)

out_cfg = {
    'experiments': experiments,
    'common': {
        'test_seq': 'processed_All_Beauty/test_seqs.json',
        'meta_file': 'meta_All_Beauty.jsonl',
        'item_pop': 'processed_All_Beauty/item_popularity.json',
        'topk': 10,
        'blair_model_path': 'blair-roberta-base-local'
    }
}
with open('eval_config_llm_only.json', 'w') as f:
    json.dump(out_cfg, f, indent=2)
"

    cp eval_config_llm_only.json eval_config.json
    python run_all_evaluations.py
    mv evaluation_results.csv results_llm_seed${seed}.csv
    ALL_CSVS="$ALL_CSVS results_llm_seed${seed}.csv"
done

# Merge into one file with seed column
python -c "
import pandas as pd, glob

dfs = []
for f in glob.glob('results_llm_seed*.csv'):
    df = pd.read_csv(f)
    df['seed'] = int(f.replace('results_llm_seed','').replace('.csv',''))
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df.to_csv('$COMBINED_CSV', index=False)

# Compute mean/std per experiment
summary = all_df.groupby('Experiment').agg(['mean','std'])
summary.to_csv('llm_multiseed_summary.csv')
print('LLM multi‑seed summary saved to llm_multiseed_summary.csv')
"

# Clean up individual seed files
rm -f results_llm_seed42.csv results_llm_seed43.csv results_llm_seed44.csv eval_config_llm_only.json
echo "Done."
