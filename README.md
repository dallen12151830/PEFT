# Parameter-Efficient Fine-Tuning with LoRA for Sentiment Classification (SST-2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dallen12151830/PEFT/blob/main/LoRA/sequence_classification.ipynb)

## 📘 Project Overview

This project demonstrates how to apply **Low-Rank Adaptation (LoRA)** for **parameter-efficient fine-tuning** (PEFT) of BERT on the **SST-2 sentiment classification** task, part of the GLUE benchmark.

Instead of full fine-tuning, LoRA updates only a small fraction of parameters, significantly reducing the computational cost while maintaining competitive performance.

---

## 🎯 Task Description

- **Dataset**: [SST-2](https://gluebenchmark.com/tasks) (binary sentiment classification)
- **Model**: `bert-base-uncased` + LoRA (PEFT)
- **Goal**: Evaluate model performance using different training data sizes (1k, 8k, 56k samples) under fixed architecture and training setup.

---

## 🧪 Experimental Results

| Training Samples | Eval Loss | Accuracy | F1-Score | Trainable Params |
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

### 1. Download the ipython file in the LoRA folder
### 2. In the file, change the folder that you want to save the checkpoint and the model weights
