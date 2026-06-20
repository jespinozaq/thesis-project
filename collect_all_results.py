import os
import csv
import re

def parse_finetuned_text(filepath):
    if not os.path.isfile(filepath): return {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    hr10 = re.search(r'HR@10:\s+([0-9.]+)', text)
    ndcg10 = re.search(r'NDCG@10:\s+([0-9.]+)', text)
    lpd = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text)
    div = re.search(r'Intra.list Diversity.*?:\s+([0-9.]+)', text)
    cov = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text)
    return {
        'HR@10': hr10.group(1) if hr10 else '',
        'NDCG@10': ndcg10.group(1) if ndcg10 else '',
        'LogPopDiff': lpd.group(1) if lpd else '',
        'Diversity': div.group(1) if div else '',
        'Coverage': cov.group(1) if cov else ''
    }

def parse_cold_start(filepath):
    """Extract cold/warm HR@10 from cold‑start evaluation file."""
    rows = []
    if not os.path.isfile(filepath): return rows
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    # Find blocks for each model
    blocks = re.split(r'----\s+(\S+)\s+----', text)
    # blocks[0] is text before first model, then alternating model name and content
    for i in range(1, len(blocks), 2):
        model = blocks[i].strip()
        content = blocks[i+1] if i+1 < len(blocks) else ''
        warm_hr = re.search(r'Warm Item Performance:.*?HR@10:\s+([0-9.]+)', content, re.DOTALL)
        cold_hr = re.search(r'Cold Item Performance:.*?HR@10:\s+([0-9.]+)', content, re.DOTALL)
        rows.append({
            'Model': model,
            'Cold_Warm_HR@10': warm_hr.group(1) if warm_hr else '',
            'Cold_Cold_HR@10': cold_hr.group(1) if cold_hr else ''
        })
    return rows

def parse_provider_fairness(filepath):
    rows = []
    if not os.path.isfile(filepath): return rows
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    blocks = re.split(r'----\s+(\S+)\s+----', text)
    for i in range(1, len(blocks), 2):
        model = blocks[i].strip()
        content = blocks[i+1] if i+1 < len(blocks) else ''
        gini = re.search(r'Gini coefficient of item exposure:\s+([0-9.]+)', content)
        hhi  = re.search(r'Herfindahl-Hirschman Index \(HHI\):\s+([0-9.]+)', content)
        rows.append({
            'Model': model,
            'Gini': gini.group(1) if gini else '',
            'HHI': hhi.group(1) if hhi else ''
        })
    return rows

def main():
    base_dir = '/workspace/thesis-project'
    all_data = []

    # 1. Main evaluation CSVs
    for fname, source in [('results_A.csv','Config A'), ('results_B.csv','Config B')]:
        path = os.path.join(base_dir, fname)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['Source'] = source
                    all_data.append(row)

    # 2. Swap experiment
    swap_path = os.path.join(base_dir, 'swap_experiment_results.csv')
    if os.path.isfile(swap_path):
        with open(swap_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['Source'] = 'Swap Experiment'
                all_data.append(row)

    # 3. Multi‑seed summary (non-LLM) 

    # 4. LLM multi‑seed summary – same, skip for now.

    # 5. Stratified recall summary
    strat_path = os.path.join(base_dir, 'stratified_recall_summary.csv')
    if os.path.isfile(strat_path):
        with open(strat_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['Source'] = 'Stratified Recall'
                all_data.append(row)

    # 6. Loss comparison (validation and test)
    for lc_file, lc_source in [('loss_comparison.csv','Loss Comparison (Val)'),
                               ('loss_comparison_test.csv','Loss Comparison (Test)')]:
        path = os.path.join(base_dir, lc_file)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['Source'] = lc_source
                    all_data.append(row)

    # 7. Cold‑start (latest file)
    cold_files = sorted([f for f in os.listdir(base_dir) if f.startswith('cold_start_low_exposure_')],
                        key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    if cold_files:
        cold_rows = parse_cold_start(os.path.join(base_dir, cold_files[0]))
        for r in cold_rows:
            r['Source'] = 'Cold‑Start (≤5)'
            all_data.append(r)

    # 8. Provider fairness (latest file)
    prov_files = sorted([f for f in os.listdir(base_dir) if f.startswith('provider_fairness_')],
                        key=lambda x: os.path.getmtime(os.path.join(base_dir, x)), reverse=True)
    if prov_files:
        prov_rows = parse_provider_fairness(os.path.join(base_dir, prov_files[0]))
        for r in prov_rows:
            r['Source'] = 'Provider Fairness'
            all_data.append(r)

    # Write master CSV
    if all_data:
        keys = set()
        for row in all_data:
            keys.update(row.keys())
        keys = sorted(keys)
        with open('thesis_master_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_data)
        print(f'thesis_master_results.csv written with {len(all_data)} rows.')
    else:
        print('No data found.')

if __name__ == '__main__':
    main()
