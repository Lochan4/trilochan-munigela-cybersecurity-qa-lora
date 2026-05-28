# Cybersecurity Q&A Fine-Tuning with LoRA

Fine-tuning **microsoft/phi-2** (2.7B) on a cybersecurity Q&A dataset using **LoRA (PEFT)** for parameter-efficient specialization. Trained on a Tesla T4 / H100 via Google Colab.

---

## 1. Dataset

**Source:** [AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0](https://huggingface.co/datasets/AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0)

**Statistics:**

| Split      | Examples |
|------------|----------|
| Train      | ~79,200  |
| Validation | ~19,800  |
| Total      | ~99,000  |

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
- Strong baseline performance on reasoning and Q&A tasks relative to its size
- Efficient inference on T4 (15GB VRAM) in bf16 with LoRA
- Active community support and well-documented architecture

**Hardware:** NVIDIA Tesla T4 (15GB VRAM) / H100 80GB (Google Colab)

**Precision:** `bfloat16` (no quantization)

**Attention:** `flash_attention_2` for memory efficiency

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

> Results are populated after running `evaluate.py` against the saved adapter.

**ROUGE Scores (validation, 10 samples):**

| Metric    | Score  |
|-----------|--------|
| rouge1    | 0.3885 |
| rouge2    | 0.1183 |
| rougeL    | 0.1831 |
| rougeLsum | 0.1824 |

**Learning curve:** see `results/learning_curve.png`

**Analysis:**
- Model adapts well to the structured `### System / Question / Answer` format
- Short factual answers score higher on ROUGE than long explanatory ones
- Occasional repetition on out-of-distribution questions (covered in `results/sample_predictions.txt`)

**Limitations:**
- Evaluation on only 10 samples due to inference latency on T4
- No quantization used; 4-bit QLoRA would reduce VRAM requirements further
- ROUGE is a surface-level metric; semantic accuracy requires human eval for cybersecurity content

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
│   ├── metrics.json
│   ├── sample_predictions.txt
│   └── training_logs.txt
├── Mitigata.ipynb      # Original Colab notebook
├── requirements.txt
├── README.md
└── .gitignore
```
