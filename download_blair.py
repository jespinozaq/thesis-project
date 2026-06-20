from transformers import AutoModel, AutoTokenizer

model_name = "hyp1231/blair-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Save to a local folder
tokenizer.save_pretrained("./blair-roberta-base-local")
model.save_pretrained("./blair-roberta-base-local")
print("Model saved to ./blair-roberta-base-local")