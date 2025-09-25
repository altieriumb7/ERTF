# load_adapter_for_inference.py
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import PeftModel
import torch

def load_model_with_adapter(base_model_name, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        device_map='auto',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='meta-llama/Llama-2-7b')
    parser.add_argument('--adapter_dir', default='outputs/erft_sft')
    args = parser.parse_args()
    tokenizer, model = load_model_with_adapter(args.base_model, args.adapter_dir)
    print('Loaded model and adapters from', args.adapter_dir)
