import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--review_file', type=str, default=None,
                        help='Path to review JSONL file (e.g., All_Beauty.jsonl). '
                             'Not required for processed-pairs training.')
    parser.add_argument('--meta_file', type=str, required=True,
                        help='Path to meta JSONL file')
    parser.add_argument('--train_pairs_file', type=str, default=None,
                        help='Path to train_pairs.jsonl (used by train_framework.py)')
    parser.add_argument('--item_pop_file', type=str, default=None,
                        help='Path to item_popularity.json (used by train_framework.py)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    # Model
    parser.add_argument('--model_name', type=str, default='roberta-base')
    parser.add_argument('--max_seq_length', type=int, default=64)
    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--cpt_epochs', type=int, default=1,
                        help='Number of CPT epochs with only IPW‑SCE and inverse‑popularity sampling')
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--beta', type=float, default=0.2,
                        help='SCE frequency scaling exponent')
    parser.add_argument('--lambda_pop', type=float, default=0.1)
    parser.add_argument('--gamma_div', type=float, default=0.05)
    parser.add_argument('--lambda_aug', type=float, default=0.1,
                        help='Weight for augmentation consistency loss')
    parser.add_argument('--topk', type=int, default=10,
                        help='Number of items to consider for diversity/popularity loss')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Use Prompt A to augment item descriptions (offline)')
    parser.add_argument('--openai_api_key', type=str, default=None)
    parser.add_argument('--augmentation_cache', type=str, default='augmented_descriptions.json')
    # Other
    parser.add_argument('--val_seq', type=str, default=None,
                        help='Path to val_seqs.json (for early stopping)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()
