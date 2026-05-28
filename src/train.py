import os
import json
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from utils import (
    MODEL_ID, DATASET_ID, SEED, MAX_LENGTH,
    load_cybersec_dataset, print_dataset_stats,
    get_tokenizer, make_texts, build_tokenize_fn,
)

OUTPUT_DIR  = "./cybersec-qlora"
ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

BATCH_SIZE    = 16
GRAD_ACCUM    = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3

LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05


class FileLoggerCallback(TrainerCallback):
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        open(path, "w").close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
        entry.update({k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()})
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Dataset
    ds = load_cybersec_dataset(DATASET_ID, seed=SEED)
    print_dataset_stats(ds)
    ds = ds.map(make_texts, desc="Formatting")

    # Tokenizer
    tokenizer    = get_tokenizer(MODEL_ID)
    tokenize_fn  = build_tokenize_fn(tokenizer, MAX_LENGTH)
    tokenized_ds = ds.map(tokenize_fn, remove_columns=ds["train"].column_names, desc="Tokenizing")
    print(f"Train sample length: {len(tokenized_ds['train'][0]['input_ids'])}")

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")

    # Training
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        optim="adamw_torch_fused",
        label_names=["labels"],
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=collator,
        callbacks=[FileLoggerCallback(f"{RESULTS_DIR}/training_logs.txt")],
    )

    print("Starting training...")
    t0 = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - t0
    h, m = divmod(int(elapsed), 3600)
    print(f"Done in {h}h {m//60}m {m%60}s")

    # Save adapter
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved → {ADAPTER_DIR}/")

    summary = {
        "model_id":        MODEL_ID,
        "dataset_id":      DATASET_ID,
        "quantization":    "none (bf16 full precision)",
        "epochs":          NUM_EPOCHS,
        "batch_size":      BATCH_SIZE,
        "effective_batch": BATCH_SIZE * GRAD_ACCUM,
        "learning_rate":   LEARNING_RATE,
        "max_length":      MAX_LENGTH,
        "lora_r":          LORA_R,
        "lora_alpha":      LORA_ALPHA,
        "train_examples":  len(tokenized_ds["train"]),
        "val_examples":    len(tokenized_ds["validation"]),
        "wall_time_sec":   round(elapsed),
    }
    with open(f"{RESULTS_DIR}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved.")


if __name__ == "__main__":
    main()
