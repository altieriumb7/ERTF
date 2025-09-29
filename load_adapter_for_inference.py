# load_adapter_for_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model_with_adapter(base_model: str, adapter_dir: str, load_in_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # <- force fp16 compute
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # <- fp16 on GPU
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = True  # speed up generation

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model
