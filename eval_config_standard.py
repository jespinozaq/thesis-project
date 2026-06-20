{
  "experiments": [
    {
      "name": "Zero-shot RoBERTa",
      "type": "zero_shot",
      "model_path": "roberta-base",
      "model_name": "roberta-base",
      "checkpoints_dir": null
    },
    {
      "name": "Zero-shot BLAIR",
      "type": "zero_shot",
      "model_path": "blair-roberta-base-local",
      "model_name": "roberta-base",
      "checkpoints_dir": null
    },
    {
      "name": "BLAIR Base (standard fine-tune, no custom loss)",
      "type": "fine_tuned",
      "model_path": "checkpoints_blair_standard_base/best_model.pt",
      "model_name": "roberta-base",
      "checkpoints_dir": "checkpoints_blair_standard_base"
    },
    {
      "name": "BLAIR Large (standard fine-tune, no custom loss)",
      "type": "fine_tuned",
      "model_path": "checkpoints_blair_standard_large/best_model.pt",
      "model_name": "roberta-base",
      "checkpoints_dir": "checkpoints_blair_standard_large"
    }
  ],
  "common": {
    "test_seq": "processed_All_Beauty/test_seqs.json",
    "meta_file": "meta_All_Beauty.jsonl",
    "item_pop": "processed_All_Beauty/item_popularity.json",
    "topk": 10,
    "blair_model_path": "blair-roberta-base-local"
  }
}
