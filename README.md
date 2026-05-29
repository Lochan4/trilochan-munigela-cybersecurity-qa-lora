# Cybersecurity Q&A Fine-Tuning with LoRA

Fine-tuning **microsoft/phi-2** (2.7B) on a cybersecurity Q&A dataset using **LoRA (PEFT)** for parameter-efficient domain specialization. Trained on NVIDIA H100 SXM (80GB).

---

## 1. Dataset

**Source:** [AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0)

**Statistics:**

| Split      | Examples |
|------------|----------|
| Train      | 79,892   |
| Validation | 19,974   |
| Total      | 99,866   |

**Columns used:** `user` (question), `assistant` (answer), `system` (optional system prompt)

**Preprocessing:**
- Filtered out rows where `user` or `assistant` is empty
- 80/20 train/validation split (seed=42)
- Formatted into a structured prompt template:
  ```
  ### System:
  <system_prompt>

  ### Question:
  <user>

  ### Answer:
  <assistant><|endoftext|>
  ```
- Prompt prefix tokens masked with `-100` in labels so loss is computed on answers only

---

## 2. Model

**Model:** [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) — 2.7B parameters

**Rationale:**
- Fits within the 3B parameter constraint
- Significantly stronger reasoning and Q&A capability than 1B-class models
- Well-documented PhiAttention + PhiMLP architecture with clear LoRA target modules
- Active community support and strong baseline on technical benchmarks

**Hardware:** NVIDIA H100 SXM (80GB VRAM)

**Precision:** `bfloat16` — H100 natively accelerates bf16, no quantization required

**Attention:** `flash_attention_2` — 2-4x speedup on 1024-token sequences

---

## 3. LoRA Configuration

| Parameter        | Value                                              |
|------------------|----------------------------------------------------|
| `r`              | 64                                                 |
| `lora_alpha`     | 128                                                |
| `lora_dropout`   | 0.05                                               |
| `bias`           | none                                               |
| `task_type`      | CAUSAL_LM                                          |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `dense`, `fc1`, `fc2` |

**Rationale:**
- `r=64` / `alpha=128` (ratio 2:1) provides strong adaptation capacity for a domain-specific task
- All attention and MLP projection layers targeted to maximise coverage of Phi-2's PhiAttention and PhiMLP blocks
- Dropout 0.05 adds light regularisation without impeding convergence
- Trainable parameters: ~84M (~3.1% of 2.7B total)

---

## 4. Training

| Hyperparameter            | Value              |
|---------------------------|--------------------|
| Epochs                    | 3                  |
| Per-device batch size     | 16                 |
| Gradient accumulation     | 2 (effective = 32) |
| Learning rate             | 2e-4               |
| LR scheduler              | cosine             |
| Warmup ratio              | 0.05               |
| Weight decay              | 0.01               |
| Max sequence length       | 1024               |
| Optimizer                 | adamw_torch_fused  |
| Precision                 | bf16               |
| Gradient checkpointing    | enabled            |
| Eval / save every         | 500 steps          |

**Actual training time:** 9h 27m on H100 80GB (3 epochs, 79,892 examples)

**Challenges:**
- Flash Attention 2 requires CUDA and a compatible GPU — falls back gracefully if unavailable
- Long answers (max ~800 words) require `max_length=1024` to avoid excessive truncation

---

## 5. Results

**ROUGE Scores (validation, 10 samples):**

| Metric    | Score  |
|-----------|--------|
| rouge1    | 0.3885 |
| rouge2    | 0.1183 |
| rougeL    | 0.1831 |
| rougeLsum | 0.1824 |

**Learning curve:** see `results/learning_curve.png`

**Overfitting analysis:**
- Train loss decreased steadily: ~2.0 → ~1.45 across 3 epochs
- Validation loss tracked closely in early training (gap < 0.01 at epoch 1), with minor divergence in later epochs — expected behaviour on a large domain-specific corpus
- Best checkpoint saved automatically via `load_best_model_at_end=True`

**Error analysis:**
- Model performs well on factual recall (CVEs, protocol definitions, OWASP categories)
- Long multi-step answers (FPGA classification, protocol design) show lower ROUGE-2 due to different but valid phrasings, not incorrect content
- Occasional repetition observed on out-of-distribution questions not well-covered by Fenrir training data
- Full predictions and references are in `results/sample_predictions.txt`

**Limitations:**
- ROUGE evaluated on 10 samples; a larger eval set would give more stable scores
- No quantization used — 4-bit QLoRA is appropriate when VRAM is constrained (e.g. T4 15GB); bf16 was chosen here as H100 80GB eliminates the need for compression
- Model may hallucinate on highly specific or recent CVEs not present in the training data
- ROUGE is a surface-level metric; semantic correctness for cybersecurity content requires human evaluation

---

## 6. Setup Instructions

```bash
# Clone the repo
git clone https://github.com/Lochan4/trilochan-munigela-cybersecurity-qa-lora.git
cd trilochan-munigela-cybersecurity-qa-lora

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Flash Attention (optional, requires CUDA)
pip install flash-attn --no-build-isolation
```

---

## 7. How to Run

### Training
```bash
cd src
python train.py
# Adapter saved to: ./cybersec-qlora/adapter/
# Logs saved to:    ./cybersec-qlora/results/training_logs.txt
```

### Evaluation
```bash
cd src
python evaluate.py --num_samples 10
# Outputs: results/metrics.json, results/sample_predictions.txt, results/learning_curve.png
```

### Inference
```bash
cd src
python inference.py "What is a SQL injection attack and how can it be prevented?"

# With custom adapter path:
python inference.py "Explain the CIA triad." --adapter_dir ./cybersec-qlora/adapter
```

---

## Project Structure

```
trilochan-munigela-cybersecurity-qa-lora/
├── src/
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script (ROUGE + learning curve)
│   ├── inference.py    # Single-question inference CLI
│   └── utils.py        # Dataset loading, prompt formatting, tokenization
├── results/
│   ├── metrics.json            # ROUGE scores
│   ├── sample_predictions.txt  # 10 prediction vs reference samples
│   ├── training_logs.txt       # Step-by-step loss logs
│   ├── training_summary.json   # Hyperparameters + wall time
│   └── learning_curve.png      # Train vs validation loss plot
├── requirements.txt
├── README.md
└── .gitignore

> **Note:** LoRA adapter weights (`adapter/`) are excluded from the repo as they exceed 100MB. Run `src/train.py` to reproduce.
```
