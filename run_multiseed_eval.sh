#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate blair_gpu
cd /workspace/thesis-project

SEEDS="42 43 44"
ALL_CSVS=""

for seed in $SEEDS; do
    echo "===== Running evaluation with seed $seed ====="

    # Update the seed for every experiment in the fast-only config
    python -c "
import json
with open('eval_config_fast.json') as f:
    cfg = json.load(f)
for exp in cfg['experiments']:
    exp['seed'] = $seed
with open('eval_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"

    python run_all_evaluations.py
    mv evaluation_results.csv results_seed${seed}.csv
    ALL_CSVS="$ALL_CSVS results_seed${seed}.csv"
done

echo "===== Combining results across seeds ====="
python -c "
import pandas as pd, glob

dfs = []
for f in glob.glob('results_seed*.csv'):
    df = pd.read_csv(f)
    df['seed'] = int(f.replace('results_seed','').replace('.csv',''))
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
summary = all_df.groupby('Experiment').agg(['mean','std'])
summary.to_csv('results_multiseed_summary.csv')
print('Multi-seed summary saved to results_multiseed_summary.csv')
"

# Clean up individual seed files
rm -f results_seed42.csv results_seed43.csv results_seed44.csv
echo "Done. Final file: results_multiseed_summary.csv"
