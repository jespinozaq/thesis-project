import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pairs', required=True)
    parser.add_argument('--meta_in', required=True)
    parser.add_argument('--meta_out', required=True)
    args = parser.parse_args()

    asins = set()
    with open(args.train_pairs, 'r', encoding='utf-8') as f:
        for line in f:
            asins.add(json.loads(line.strip())['asin'])

    print(f"Found {len(asins)} unique ASINs in training pairs.")
    with open(args.meta_in, 'r', encoding='utf-8') as fin, \
         open(args.meta_out, 'w', encoding='utf-8') as fout:
        for line in fin:
            item = json.loads(line.strip())
            if item['parent_asin'] in asins:
                fout.write(line)
    print(f"Filtered meta written to {args.meta_out}.")

if __name__ == '__main__':
    main()
