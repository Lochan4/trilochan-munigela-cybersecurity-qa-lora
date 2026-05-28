import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

DATASET_ID = "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0"
MODEL_ID   = "microsoft/phi-2"
EOS_TOKEN  = "<|endoftext|>"
MAX_LENGTH = 1024
SEED       = 42

SYSTEM_PROMPT = (
    "You are a highly specialized AI assistant for advanced cyber-defense. "
    "Deliver accurate, in-depth, actionable guidance on information-security "
    "principles including confidentiality, integrity, availability, "
    "authenticity, non-repudiation, and privacy."
)


def load_cybersec_dataset(dataset_id: str = DATASET_ID, seed: int = SEED):
    raw   = load_dataset(dataset_id)
    full  = raw["train"].filter(lambda ex: bool(ex["user"]) and bool(ex["assistant"]))
    split = full.train_test_split(test_size=0.2, seed=seed)
    ds    = DatasetDict({"train": split["train"], "validation": split["test"]})
    return ds


def print_dataset_stats(ds: DatasetDict):
    print(f"Train      : {len(ds['train'])} examples")
    print(f"Validation : {len(ds['validation'])} examples")
    a_lens = [len(ex["assistant"].split()) for ex in ds["train"]]
    print(f"Answer words — min:{min(a_lens)} max:{max(a_lens)} mean:{np.mean(a_lens):.0f}")


def format_prompt(question: str, answer: str = None, system: str = None) -> str:
    sys_text = system if system else SYSTEM_PROMPT
    prompt   = f"### System:\n{sys_text}\n\n### Question:\n{question}\n\n### Answer:\n"
    if answer is not None:
        prompt += answer + EOS_TOKEN
    return prompt


def make_texts(example):
    return {
        "full_text":   format_prompt(example["user"], example["assistant"], example.get("system")),
        "prefix_text": format_prompt(example["user"], answer=None,          system=example.get("system")),
    }


def get_tokenizer(model_id: str = MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    return tokenizer


def build_tokenize_fn(tokenizer, max_length: int = MAX_LENGTH):
    def tokenize(example):
        full_enc   = tokenizer(example["full_text"],   truncation=True,  max_length=max_length, padding=False)
        prefix_enc = tokenizer(example["prefix_text"], truncation=False, padding=False)

        input_ids  = full_enc["input_ids"]
        labels     = list(input_ids)
        prefix_len = len(prefix_enc["input_ids"])
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": full_enc["attention_mask"],
            "labels":         labels,
        }
    return tokenize
