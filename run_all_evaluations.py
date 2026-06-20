import subprocess, re, json, csv, os, sys
from collections import OrderedDict

def parse_sequential_output(output):
    res = {}
    for line in output.splitlines():
        m = re.search(r'HR@(\d+):\s+([0-9.]+)', line)
        if m: res[f'HR@{m.group(1)}'] = float(m.group(2))
        m = re.search(r'NDCG@(\d+):\s+([0-9.]+)', line)
        if m: res[f'NDCG@{m.group(1)}'] = float(m.group(2))
        m = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', line)
        if m: res['LogPopDiff'] = float(m.group(1))
        m = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', line)
        if m: res['Diversity'] = float(m.group(1))
        m = re.search(r'Catalogue Coverage:\s+([0-9.]+)', line)
        if m: res['Coverage'] = float(m.group(1))
    return res

def parse_llm_output(output):
    res = {}
    for k in [1, 5, 10]:
        m = re.search(rf'HR@{k}:\s+([0-9.]+)', output)
        if m: res[f'HR@{k}'] = float(m.group(1))
        m = re.search(rf'NDCG@{k}:\s+([0-9.]+)', output)
        if m: res[f'NDCG@{k}'] = float(m.group(1))
    m = re.search(r'Average Log Popularity Difference:\s+([0-9.\-]+)', output)
    if m: res['LogPopDiff'] = float(m.group(1))
    m = re.search(r'Intra-list Diversity.*?:\s+([0-9.]+)', output)
    if m: res['Diversity'] = float(m.group(1))
    m = re.search(r'Catalogue Coverage:\s+([0-9.]+)', output)
    if m: res['Coverage'] = float(m.group(1))
    return res

def run_cmd(cmd_args, cwd=None):
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(cmd_args, capture_output=True, text=True, cwd=cwd)
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        print(f"ERROR running {' '.join(cmd_args[:2])}: {combined[-500:]}")
    return combined

def main():
    config_file = "eval_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    common = config['common']
    experiments = config['experiments']
    results = []

    python_exe = sys.executable

    for exp in experiments:
        name = exp['name']
        exp_type = exp['type']
        print(f"\n===== Running: {name} =====")

        if exp_type == 'zero_shot':
            cmd = [
                python_exe, "evaluate_sequential.py",
                "--model_path", exp['model_path'],
                "--model_name", exp['model_name'],
                "--test_seq", common['test_seq'],
                "--meta_file", common['meta_file'],
                "--item_pop", common['item_pop'],
                "--topk", str(common['topk'])
            ]
            out = run_cmd(cmd)
            metrics = parse_sequential_output(out)

        elif exp_type == 'fine_tuned':
            eval_script = "evaluate_finetuned.py"
            if 'llama' in exp.get('model_name', '').lower():
                eval_script = "evaluate_llm_finetuned.py"
            cmd = [
                python_exe, eval_script,
                "--model_path", exp['model_path'],
                "--model_name", exp['model_name'],
                "--test_seq", common['test_seq'],
                "--meta_file", common['meta_file'],
                "--item_pop", common['item_pop'],
                "--topk", str(common['topk'])
            ]
            out = run_cmd(cmd)
            metrics = parse_sequential_output(out)

        elif exp_type == 'llm':
            cmd = [
                python_exe, "evaluate_llm_baselines.py",
                "--test_seqs", common['test_seq'],
                "--meta_file", common['meta_file'],
                "--item_pop", common['item_pop'],
                "--backend", exp['backend'],
                "--model", exp['model'],
                "--prompt_mode", exp['prompt_mode'],
                "--topk", str(common['topk']),
                "--blair_model_path", common['blair_model_path']
            ]
            if 'base_url' in exp: cmd.extend(['--base_url', exp['base_url']])
            env_key = exp.get('api_key_env', 'OPENAI_API_KEY')
            key = os.environ.get(env_key, '')
            if key: cmd.extend(['--api_key', key])
            # Add parallelism args from config (with defaults)
            nw = exp.get('num_workers', 1)
            rd = exp.get('request_delay', 0.5)
            cmd.extend(['--num_workers', str(nw), '--request_delay', str(rd)])
            out = run_cmd(cmd)
            metrics = parse_llm_output(out)

        elif exp_type == 'combined':
            cmd = [
                python_exe, "rerank_promptB.py",
                "--model_path", exp['model_path'],
                "--model_name", exp['model_name'],
                "--test_seq", common['test_seq'],
                "--meta_file", common['meta_file'],
                "--item_pop", common['item_pop'],
                "--backend", exp['backend'],
                "--llm_model", exp['llm_model'],
                "--candidate_pool", "100",
                "--topk", str(common['topk'])
            ]
            if 'base_url' in exp: cmd.extend(['--base_url', exp['base_url']])
            env_key = exp.get('api_key_env', 'OPENAI_API_KEY')
            key = os.environ.get(env_key, '')
            if key: cmd.extend(['--api_key', key])
            nw = exp.get('num_workers', 1)
            rd = exp.get('request_delay', 0.5)
            cmd.extend(['--num_workers', str(nw), '--request_delay', str(rd)])
            out = run_cmd(cmd)
            metrics = parse_llm_output(out)

        else:
            print(f"Unknown type {exp_type}")
            continue

        row = OrderedDict([
            ('Experiment', name),
            ('HR@1', metrics.get('HR@1', None)),
            ('HR@5', metrics.get('HR@5', None)),
            ('HR@10', metrics.get('HR@10', None)),
            ('NDCG@10', metrics.get('NDCG@10', None)),
            ('LogPopDiff', metrics.get('LogPopDiff', None)),
            ('Diversity', metrics.get('Diversity', None)),
            ('Coverage', metrics.get('Coverage', None))
        ])
        results.append(row)
        print("Collected:", row)

    csv_path = "evaluation_results.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nAll results saved to {csv_path}")

if __name__ == '__main__':
    main()
