import os
import json
import argparse

import torch
import evaluate as ev
import matplotlib.pyplot as plt
from peft import PeftModel
from transformers import AutoModelForCausalLM

from utils import (
    MODEL_ID, SEED,
    load_cybersec_dataset, get_tokenizer,
    make_texts, format_prompt,
)

OUTPUT_DIR  = "./cybersec-qlora"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
RESULTS_DIR = f"{OUTPUT_DIR}/results"
NUM_SAMPLES = 10


def load_model_and_tokenizer(adapter_dir: str = ADAPTER_DIR):
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


def run_rouge_eval(model, tokenizer, ds, num_samples: int = NUM_SAMPLES):
    rouge     = ev.load("rouge")
    val_raw   = ds["validation"].select(range(min(num_samples, len(ds["validation"]))))
    preds, refs, lines = [], [], []

    for i, ex in enumerate(val_raw):
        prompt = format_prompt(ex["user"], answer=None, system=ex.get("system"))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        preds.append(decoded)
        refs.append(ex["assistant"])
        lines += [
            f"=== SAMPLE {i+1} ===",
            f"Q: {ex['user'][:120]}",
            f"PRED: {decoded[:300]}",
            f"REF : {ex['assistant'][:300]}",
            "",
        ]

    scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    return scores, lines


def plot_learning_curve(logs_path: str, output_path: str):
    logs = []
    with open(logs_path) as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except Exception:
                pass

    train_steps = [l["step"] for l in logs if "loss" in l and "eval_loss" not in l]
    train_loss  = [l["loss"]  for l in logs if "loss" in l and "eval_loss" not in l]
    eval_steps  = [l["step"]  for l in logs if "eval_loss" in l]
    eval_loss   = [l["eval_loss"] for l in logs if "eval_loss" in l]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_loss, label="Train loss", alpha=0.8)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="Val loss", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss — Cybersecurity QA LoRA")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", default=ADAPTER_DIR)
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    ds = load_cybersec_dataset()
    ds = ds.map(make_texts, desc="Formatting")

    model, tokenizer = load_model_and_tokenizer(args.adapter_dir)

    print(f"Running ROUGE eval on {args.num_samples} validation samples...")
    scores, lines = run_rouge_eval(model, tokenizer, ds, args.num_samples)

    print("ROUGE scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    with open(f"{args.results_dir}/sample_predictions.txt", "w") as f:
        f.write("\n".join(lines))

    scores["num_samples"] = args.num_samples
    with open(f"{args.results_dir}/metrics.json", "w") as f:
        json.dump({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()}, f, indent=2)

    print(f"Saved metrics.json and sample_predictions.txt → {args.results_dir}/")

    logs_path = f"{OUTPUT_DIR}/results/training_logs.txt"
    if os.path.exists(logs_path):
        plot_learning_curve(logs_path, f"{args.results_dir}/learning_curve.png")


if __name__ == "__main__":
    main()
