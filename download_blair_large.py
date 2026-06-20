from transformers import AutoModel, AutoTokenizer

model_name = "hyp1231/blair-roberta-large"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)  # actually roberta-large tokenizer
model.save_pretrained("./blair-roberta-large-local")
tokenizer.save_pretrained("./blair-roberta-large-local")