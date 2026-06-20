import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

class LLMEncoder(nn.Module):
    """Encodes text with a quantised LLM + LoRA, returning L2‑normalised embedding."""
    def __init__(self, model_name='meta-llama/Llama-3.1-8B-Instruct', pooling='last'):
        super().__init__()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.transformer = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.float16
        )
        for param in self.transformer.parameters():
            param.requires_grad = False
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['q_proj', 'v_proj']
        )
        self.transformer = get_peft_model(self.transformer, peft_config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1]
        if self.pooling == 'last':
            lengths = attention_mask.sum(dim=1) - 1
            embeddings = last_hidden[torch.arange(last_hidden.size(0)), lengths]
        else:
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_emb = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_emb / sum_mask
        return F.normalize(embeddings, p=2, dim=-1)

class LLMRecommender(nn.Module):
    """Two‑tower architecture with two independent LLM encoders."""
    def __init__(self, model_name='meta-llama/Llama-3.1-8B-Instruct'):
        super().__init__()
        self.context_encoder = LLMEncoder(model_name)
        self.item_encoder = LLMEncoder(model_name)

    def encode_context(self, input_ids, attention_mask):
        return self.context_encoder(input_ids, attention_mask)

    def encode_item(self, input_ids, attention_mask):
        return self.item_encoder(input_ids, attention_mask)
