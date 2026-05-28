import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from utils import MODEL_ID, format_prompt, get_tokenizer

OUTPUT_DIR  = "./cybersec-qlora"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"


def load_model(adapter_dir: str = ADAPTER_DIR):
    tokenizer = get_tokenizer(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def answer(question: str, model, tokenizer, max_new_tokens: int = 256) -> str:
    prompt = format_prompt(question, answer=None)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity Q&A inference")
    parser.add_argument("question", type=str, help="The cybersecurity question to answer")
    parser.add_argument("--adapter_dir", default=ADAPTER_DIR)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.adapter_dir)

    print(f"\nQ: {args.question}\n")
    response = answer(args.question, model, tokenizer, args.max_new_tokens)
    print(f"A: {response}")


if __name__ == "__main__":
    main()
