# Parameter-Efficient Fine-Tuning with LoRA for Sentiment Classification (SST-2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dallen12151830/PEFT/blob/main/LoRA/sequence_classification.ipynb)

## 📘 Project Overview

This project demonstrates how to apply **Low-Rank Adaptation (LoRA)** for **parameter-efficient fine-tuning** (PEFT) of BERT on the **SST-2 sentiment classification** task, part of the GLUE benchmark.

Instead of full fine-tuning, LoRA updates only a small fraction of parameters, significantly reducing the computational cost while maintaining competitive performance.

---

## 🎯 Task Description

- **Dataset**: SST-2 (binary sentiment classification)
- **Model**: `bert-base-uncased` fine-tuned using LoRA (Low-Rank Adaptation)
- **PEFT Method**: [LoRA](https://arxiv.org/abs/2106.09685) – Parameter-Efficient Fine-Tuning
- **LoRA Configuration**:
  ```python
  LoraConfig(
      r=8,
      lora_alpha=16,
      lora_dropout=0.1,
      bias="none",
      task_type=TaskType.SEQ_CLS,
      target_modules=["query", "value"]
  )
  ```
- **Trainable Parameters**: Only 0.27% of total model parameters are updated during training.
- **Goal**: Evaluate model performance using different training data sizes (1k, 8k, 56k samples) under fixed architecture and training setup.

---

## 🧪 Experimental Results

| Training Samples | Test Loss | Accuracy | F1-Score | Trainable Params |
|------------------|-----------|----------|----------|------------------|
| 1,000            | 0.329     | 87.3%    | 0.875    | 0.27%            |
| 8,000            | 0.240     | 90.6%    | 0.906    | 0.27%            |
| 56,000           | 0.172     | 93.5%    | 0.934    | 0.27%            |

> 🔍 **Observation**: As training size increases, both accuracy and F1-score consistently improve. LoRA-based fine-tuning achieves the **same 93.5% accuracy as full fine-tuning** (reported in original BERT paper) with only **0.27% of parameters updated**.

---

## 📈 Visual Outputs

- Confusion Matrix
- Accuracy/Loss Curve per Epoch
- Training Logs saved as CSV
- Output Plots saved as `loss.pdf` and `acc.pdf`

These outputs can be reproduced by running the notebook in Colab or locally.

---

## ⚙️ Setup Instructions

### 1. Download the IPython file (sequence_classification.ipynb) from the LoRA/ folder.
### 2. Update the working directory in the notebook:
Modify this line to specify where to save checkpoints and model weights:
  ```python
  working_dir = '/your/custom/folder/path'
  ```
### 3. Run the notebook in Colab or Jupyter:
All dependencies will be installed automatically (e.g., transformers, datasets, peft).
