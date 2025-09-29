# load_adapter_for_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_model_with_adapter(base_model: str, adapter_dir: str, load_in_4bit: bool = True):
    """
    Load a plain AutoModelForCausalLM and attach a PEFT LoRA adapter.
    Using BitsAndBytesConfig to avoid deprecated load_in_4bit/load_in_8bit args.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        # Llama-2 typically has no pad_token; use eos as pad to keep generate() happy.
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
    )

    # Attach LoRA
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return tokenizer, model
