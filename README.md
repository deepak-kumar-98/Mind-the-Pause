# Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs

> **ACL 2026** | Deepak Kumar, Baban Gain, Asif Ekbal | Indian Institute of Technology, Patna

📧 deepakkumar1538@gmail.com · gainbaban@gmail.com · asif@iitp.ac.in

---

## 📘 Overview

Spontaneous speech often includes fillers, repetitions, false starts, and discourse markers. These disfluencies appear in ASR transcripts, reducing readability and hindering downstream applications such as chatbots, voice assistants, and dialogue systems.

**Mind-the-Pause** introduces the **first LLM-based multilingual disfluency correction pipeline** for Indic languages, integrating:

- 🔍 **MuRIL-based** token-level disfluency detection (fluent / disfluent tagging)
- 🧠 **Instruction-tuned LLM correction** via LLaMA-3.2-3B-Instruct, conditioned on MuRIL-generated labels
- 🎯 **Contrastive learning objective** that explicitly penalizes regeneration of disfluent tokens

Evaluated on **Hindi**, **Bengali**, and **Marathi**, the proposed approach achieves:

- **+1.97 BLEU** over non-contrastive training
- **+6.16 BLEU** over multilingual instruction fine-tuning
- **+8.54 BLEU** over mBART baseline
- Matches or surpasses **GPT-4o** on 4 out of 6 evaluation conditions
- Substantially outperforms **Gemini 2.5 Pro** across all three languages

---

## 🏗️ Repository Structure

```
MIND-THE-PAUSE/
│
├── muril_detection/           # Disfluency detection using MuRIL (token classification)
├── with_contrastive_loss/     # Instruction tuning + contrastive objective
├── without_contrastive_ft/    # Instruction tuning without contrastive loss
├── mBART/                     # Encoder-decoder baseline (mBART fine-tuning)
├── zero_shot_prompting/       # Zero-shot and few-shot LLM prompting experiments
├── llm_as_a_judge/            # LLM-based evaluation setup for fluency and adequacy
│
├── requirements.txt           # Dependencies
└── README.md
```

---

## ⚙️ Core Components

### 🧩 1. MuRIL-Based Detection

Fine-tunes [MuRIL](https://huggingface.co/google/muril-base-cased) — a transformer encoder pretrained on 17 Indian languages — for token classification. Each token is labelled as **fluent (0)** or **disfluent (1)**. The model is trained jointly on Hindi, Bengali, and Marathi using standard cross-entropy loss over token labels:

```
L_detect = - Σ Σ log P_θ(y_t | x_i)
```

On manually edited data, MuRIL achieves F1 scores of **0.988 (Hindi)**, **0.987 (Bengali)**, and **0.986 (Marathi)**.

### 🧠 2. Instruction-Tuned Correction

Uses **LLaMA-3.2-3B-Instruct** in an Alpaca-style instruction–input–output format. Each training instance supplies:
- The disfluent ASR transcript
- Token-level fluent/disfluent labels predicted by MuRIL
- The target fluent reference sentence

This conditions the LLM not only to drop disfluent tokens but to regenerate a well-formed fluent sentence.

### 🎯 3. Contrastive Anti-Disfluency Objective

Introduces a novel multi-loss objective combining standard cross-entropy with a **contrastive penalty** that explicitly discourages the model from regenerating known disfluent tokens:

```
L_total = L_CE + λ · L_contrastive
```

The contrastive term uses geometric decay weights for disfluent sub-tokens and is applied with warm-up scheduling (λ = 0.3). This push–pull dynamic aligns outputs with fluent targets while actively suppressing disfluency patterns across languages.

### 💬 4. LLM-as-a-Judge Evaluation

Employs **Qwen2.5-3B-Instruct** to automatically assess model outputs for **fluency** and **meaning preservation** via pairwise comparison. Qwen is chosen over LLaMA to mitigate self-preference bias; comparisons are run in both directions and averaged to reduce positional bias.

---

## 📊 Dataset

We use the parallel fluent–disfluent dataset from [Kundu et al. (2022)](https://aclanthology.org/2022.coling-1.393) for Hindi, Bengali, and Marathi — the only publicly available parallel resource for disfluency correction in these languages.

| Language | Synthetic Pairs | Manually Edited Test | Real Test |
|---|---|---|---|
| Hindi | ~40,000 | 575 sentences | 150 sentences |
| Bengali | ~40,000 | 500 sentences | 300 sentences |
| Marathi | ~40,000 | 420 sentences | 250 sentences |
| **Total** | **~120,000** | **1,495 sentences** | **700 sentences** |

Train/validation/test splits follow an **80–10–10** strategy for both multilingual and monolingual settings.

---

## 📈 Main Results

### LLaMA-3.2-3B-Instruct (BLEU ↑)

| Language | Dataset | Multilingual FT | mBART | W/o Contrastive | **With Contrastive** |
|---|---|---|---|---|---|
| Hindi | Manually Edited | 92.5 | 85.7 | 93.8 | **94.6** |
| Hindi | Real Data | 64.8 | 71.4 | 87.4 | **90.4** |
| Bengali | Manually Edited | 94.2 | 84.3 | 92.6 | **94.8** |
| Bengali | Real Data | 69.6 | 73.5 | 70.7 | **74.4** |
| Marathi | Manually Edited | 94.4 | 83.7 | 93.0 | **94.7** |
| Marathi | Real Data | 80.0 | 82.6 | 83.2 | **83.6** |

### Qwen-2.5-3B-Instruct with Contrastive Loss

| Language | Dataset | BLEU ↑ | chrF2 ↑ | TER ↓ |
|---|---|---|---|---|
| Hindi | Manually Edited | **96.1** | **98.0** | **3.7** |
| Hindi | Real Data | **91.1** | **94.9** | **6.0** |
| Bengali | Manually Edited | **96.4** | **98.6** | **3.4** |
| Bengali | Real Data | **75.9** | **93.2** | **16.2** |
| Marathi | Manually Edited | **95.1** | **98.4** | **3.8** |
| Marathi | Real Data | **84.4** | **94.3** | **10.0** |

### vs. Frontier Models (Real Data, Zero-shot)

| Model | Hindi BLEU | Bengali BLEU | Marathi BLEU |
|---|---|---|---|
| GPT-4o (few-shot) | 87.2 | 79.6 | 86.9 |
| Gemini 2.5 Pro (zero-shot) | 70.5 | 61.8 | 74.9 |
| **Ours (Qwen, Contrastive)** | **91.1** | **75.9** | **84.4** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (tested on single NVIDIA A100 80 GB)
- Hugging Face access to [LLaMA-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

### Installation

```bash
git clone https://github.com/deepak-kumar-98/Mind-the-Pause.git
cd Mind-the-Pause
pip install -r requirements.txt
```

### Training & Evaluation

```bash
# Step 1: Train MuRIL for token-level disfluency detection
cd muril_detection
python train_muril.py

# Step 2: Fine-tune LLaMA with contrastive loss (proposed method)
cd ../with_contrastive_loss
python train_contrastive.py

# Step 3: Fine-tune LLaMA without contrastive loss (ablation)
cd ../without_contrastive_ft
python train_ft.py

# Step 4: Run mBART encoder-decoder baseline
cd ../mBART
python train_mbart.py

# Step 5: Zero-shot / few-shot prompting experiments
cd ../zero_shot_prompting
python run_prompting.py

# Step 6: Evaluate with automatic metrics (BLEU / chrF2 / TER)
python evaluate.py

# Step 7: LLM-as-a-Judge pairwise evaluation
cd ../llm_as_a_judge
python judge_eval.py
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Sequence length | 512 tokens |
| Effective batch size | 16 (gradient accumulation × 2, per-device batch size 8) |
| Learning rate | 1e-5 (cosine decay, warmup ratio 0.1) |
| Contrastive weight λ | 0.3 (with warm-up) |
| Label smoothing | 0.01 |
| Precision | bfloat16 |
| Optimizer | Fused AdamW |
| Early stopping patience | 3 epochs |
| Hardware | Single NVIDIA A100 (80 GB) |

---

## 💬 Instruction Prompt (Contrastive Method)

```
You are given a disfluent sentence generated by an Automatic Speech Recognition (ASR) system.
The sentence may contain disfluencies such as repetitions, fillers (e.g., 'um', 'uh'),
discourse markers (e.g., 'you know', 'I mean'), or false starts in {Language}.

Your task is to remove these disfluencies while preserving the original meaning
and grammatical correctness.

You are also provided with:
- The disfluent sentence
- A tokenized version of the sentence
- A sequence of predicted labels for each token, where:
  • '1' = the token is disfluent and should be removed
  • '0' = the token is fluent and should be retained
- A list of disfluent tokens that must be removed from the sentence

Using this information, reconstruct the fluent sentence.

Tokenized Input: {tokens}
Predicted Labels: {labels}
Disfluent Sentence: {disfluent}
Disfluent Tokens: {disfluent_tokens}
Fluent Sentence:
```

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@misc{kumar2026mindpausedisfluencyawareobjective,
      title={Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs}, 
      author={Deepak Kumar and Baban Gain and Asif Ekbal},
      year={2026},
      eprint={2605.12242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.12242}, 
}
```

---

## 🙏 Acknowledgement

The authors gratefully acknowledge the **COIL-D (Centre of Indian Language Data) Project** under Bhashini, funded by **MeitY, Government of India**, for providing the support and resources that enabled this research.

---

## 📜 License

| Component | License |
|---|---|
| Qwen2.5-3B-Instruct | Qwen Research License Agreement |
| LLaMA-3.2-3B-Instruct | Llama 3.2 Community License |
| MuRIL | Apache 2.0 |
| Code (this repo) | MIT License |
