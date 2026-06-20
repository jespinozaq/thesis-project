import os, csv, re, json

BASE = "/workspace/thesis-project"
APPLIANCES = os.path.join(BASE, "appliances_results")

def parse_finetuned_text(filepath):
    """Extract metrics from evaluate_finetuned.py output."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    hr10 = re.search(r'HR@10:\s+([0-9.]+)', text)
    ndcg10 = re.search(r'NDCG@10:\s+([0-9.]+)', text)
    lpd = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', text)
    div = re.search(r'Intra.list Diversity.*?:\s+([0-9.]+)', text)
    cov = re.search(r'Catalogue Coverage:\s+([0-9.]+)', text)
    return {
        'HR@10': hr10.group(1) if hr10 else 'N/A',
        'NDCG@10': ndcg10.group(1) if ndcg10 else 'N/A',
        'LogPopDiff': lpd.group(1) if lpd else 'N/A',
        'Diversity': div.group(1) if div else 'N/A',
        'Coverage': cov.group(1) if cov else 'N/A'
    }

def print_table(title, rows, headers):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    if not rows:
        print("  No data found.")
        return
    # Determine column widths
    col_widths = [max(len(str(row.get(h, ''))) for row in rows) for h in headers]
    col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]
    # Header
    header_line = " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers))
    print("  " + header_line)
    print("  " + "-" * len(header_line))
    # Rows
    for row in rows:
        vals = [str(row.get(h, '')).center(col_widths[i]) for i, h in enumerate(headers)]
        print("  " + " | ".join(vals))
    print()

def main():
    # ------ Beauty: results_A.csv, results_B.csv ------
    beauty_main = []
    for fname, label in [("results_A.csv", "Config A"), ("results_B.csv", "Config B")]:
        path = os.path.join(BASE, fname)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['Source'] = label
                    beauty_main.append(row)
    if beauty_main:
        headers = ['Experiment','HR@10','NDCG@10','LogPopDiff','Diversity','Coverage','Source']
        print_table("Beauty – Main Evaluations (Config A & B)", beauty_main, headers)

    # ------ Beauty: swap_experiment_results.csv ------
    swap_path = os.path.join(BASE, "swap_experiment_results.csv")
    swap_rows = []
    if os.path.isfile(swap_path):
        with open(swap_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                swap_rows.append(row)
    if swap_rows:
        headers = ['Experiment','HR@10','NDCG@10','LogPopDiff','Diversity','Coverage']
        print_table("Beauty – Swap Experiment (pool=30)", swap_rows, headers)

    # ------ Beauty: multi-seed (non-LLM) ------
    multiseed_path = os.path.join(BASE, "results_multiseed_summary.csv")
    if os.path.isfile(multiseed_path):
        with open(multiseed_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        if len(lines) >= 2:
            # lines[0] is multi-level header, lines[1] is sub-header; data from line 2 onward
            print(f"\n{'='*80}")
            print("  Beauty – Multi‑Seed Summary (non‑LLM)")
            print(f"{'='*80}")
            for line in lines:
                print("  " + " | ".join(line))
            print()

    # ------ Beauty: LLM multi-seed ------
    llm_ms_path = os.path.join(BASE, "llm_multiseed_summary.csv")
    if os.path.isfile(llm_ms_path):
        with open(llm_ms_path, 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
        if len(lines) >= 2:
            print(f"\n{'='*80}")
            print("  Beauty – LLM Multi‑Seed Summary (means ± std)")
            print(f"{'='*80}")
            for line in lines:
                print("  " + " | ".join(line))
            print()

    # ------ Beauty: stratified recall ------
    strat_path = os.path.join(BASE, "stratified_recall_summary.csv")
    if os.path.isfile(strat_path):
        with open(strat_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        headers = ['model','head_HR','mid_HR','tail_HR']
        print_table("Beauty – Stratified Recall (head/mid/tail)", rows, headers)

    # ------ Beauty: loss comparison ------
    loss_path = os.path.join(BASE, "loss_comparison.csv")
    if os.path.isfile(loss_path):
        with open(loss_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            headers = ['loss','best_epoch','val_hr','val_ndcg','val_popdiff','val_diversity']
            print_table("Beauty – Loss Comparison (Validation Metrics)", rows, headers)
    loss_test_path = os.path.join(BASE, "loss_comparison_test.csv")
    if os.path.isfile(loss_test_path):
        with open(loss_test_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            headers = ['loss','test_hr@10','test_ndcg@10','test_popdiff','test_diversity','test_coverage']
            print_table("Beauty – Loss Comparison (Test Metrics, Top‑3 Losses)", rows, headers)

    # ---------- Appliances ----------
    appliances_rows = []

    # 1. Zero-shot BLAIR
    zero = parse_finetuned_text(os.path.join(APPLIANCES, "zeroshot_blair_metrics.txt"))
    if zero:
        zero['Experiment'] = 'Zero‑shot BLAIR (Appliances)'
        appliances_rows.append(zero)

    # 2. Fine-tuned base
    base = parse_finetuned_text(os.path.join(APPLIANCES, "base_metrics.txt"))
    if base:
        base['Experiment'] = 'BLAIR Base (full aug, Appliances)'
        appliances_rows.append(base)

    # 3. ChatGPT fair baseline
    chat = parse_finetuned_text(os.path.join(APPLIANCES, "chatgpt_fair.txt"))
    if chat:
        chat['Experiment'] = 'ChatGPT fair (Appliances)'
        appliances_rows.append(chat)

    # 4. DeepSeek fair baseline
    ds = parse_finetuned_text(os.path.join(APPLIANCES, "deepseek_fair.txt"))
    if ds:
        ds['Experiment'] = 'DeepSeek fair (Appliances)'
        appliances_rows.append(ds)

    # 5. Swap combined (the script saved swap_llm.txt and swap_baseline.txt)
    swap_base_file = os.path.join(APPLIANCES, "swap_baseline.txt")
    swap_llm_file  = os.path.join(APPLIANCES, "swap_llm.txt")
    swap_base = parse_finetuned_text(swap_base_file)
    if swap_base:
        swap_base['Experiment'] = 'Retriever Top‑10 (pool=30 baseline, Appliances)'
        appliances_rows.append(swap_base)
    swap_llm = parse_finetuned_text(swap_llm_file)
    if swap_llm:
        swap_llm['Experiment'] = 'Swap Combined (pool=30, keep=8, swap=2, Appliances)'
        appliances_rows.append(swap_llm)

    if appliances_rows:
        headers = ['Experiment','HR@10','NDCG@10','LogPopDiff','Diversity','Coverage']
        print_table("Appliances – Results", appliances_rows, headers)

    # Stratified recall for Appliances (only base model)
    app_strat_file = os.path.join(BASE, "recommendations_appliances_base_fullaug.json")
    # If we want to compute it again we could, but we'll note where to find it.
    if os.path.isfile(app_strat_file):
        print(f"\n  Appliances stratified recall data is available in: {app_strat_file}")
        print("  (You can re‑run the stratified recall loop for Appliances to get a summary CSV.)\n")

    # Qualitative examples are text files; just note their location
    print(f"  Qualitative examples (Beauty): {os.path.join(BASE, 'qualitative_examples.txt')}")
    print(f"  Qualitative examples (Appliances): {os.path.join(APPLIANCES, 'qualitative.txt')}")

if __name__ == '__main__':
    main()
