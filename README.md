# Personality Inference from Semi-Structured Interviews using Large Language Models

🚧 Under Review

---

## 🔍 Overview

This repository contains code and resources for our study on **personality trait prediction using large language models (LLMs)** from **semi-structured interview transcripts**.

We systematically evaluate four modeling paradigms:

- Prompt-based inference (GPT-5 Mini; zero-shot and chain-of-thought)
- Parameter-efficient fine-tuning (LoRA; RoBERTa and Meta-LLaMA)
- Supervised fine-tuning (GPT-4.1 Mini)
- Embedding-based regression (MPNet, E5, OpenAI embeddings)

---

## 📊 Key Findings

- Prompt-based methods show **weak and unstable performance**
- Supervised fine-tuning achieves **strong correlations (r ≈ 0.7)**
- Embedding-based regression provides **competitive performance**
- Prediction performance varies substantially across traits:
  - Conscientiousness & Agreeableness → more predictable  
  - Openness → consistently difficult  

---

## 📂 Dataset

We use a real-world dataset of:

- **N = 518 participants**
- Semi-structured interview transcripts (~15 minutes each)
- Ground-truth Big Five personality scores (derived from BFI-10)

📎 Dataset available via OSF:  
https://osf.io/357yk/overview?view_only=12566422d5394a6eb78b89e6e597fe01

---

## 🧠 Prompt Design

### Zero-shot Prompt
Direct inference of Big Five trait scores from interview transcripts.

### Chain-of-Thought (CoT) Prompt
Structured reasoning:
1. Extract behavioral evidence  
2. Reason about each trait  
3. Assign continuous scores (1–5)

Full templates are provided in `/prompts/`.

---

## ⚙️ Methods

### 1. Prompt-based Inference
- Model: GPT-5 Mini  
- Default inference settings  
- Structured output format: `Trait: X.X`

---

### 2. LoRA Fine-tuning
- Models: RoBERTa-base, LLaMA-2  
- Rank settings: 8 / 16 / 32  
- Chunk size: 512 tokens  

---

### 3. Supervised Fine-tuning
- Model: GPT-4.1 Mini  
- Train/test split: 80/20  
- Multiple independent runs + averaging  

---

### 4. Embedding-based Regression
- Encoders:
  - all-mpnet-base-v2  
  - E5-large-v2  
  - text-embedding-3-small  
  - text-embedding-3-large  
- Model: Multi-output Ridge regression (α = 1.0)

---

## 📈 Evaluation

We report:

- Pearson correlation (r)  
- Mean Absolute Error (MAE)  
- Statistical significance (p-values, confidence intervals)

A naive baseline (mean predictor) is included for comparison.

---

## ⚠️ Limitations

- BFI-10 introduces a reliability ceiling (α ≈ 0.65–0.75)  
- Long transcripts require chunking, which may dilute signals  
- Sample size (N = 518) limits generalization  

---

## 🔮 Future Work

- Multimodal personality inference (text + video + audio)  
- Hierarchical encoding for long-form conversations  
- Domain-adapted LLMs for psychological tasks  

---

## 📁 Repository Structure
.
├── prompts/ # Prompt templates
├── src/ # Code for experiments
├── data/ # Dataset description (link to OSF)
├── results/ # Tables and outputs
├── paper/ # Manuscript

---

## 📜 Citation

If you find this work useful, please cite:


## 📬 Contact

For questions or collaboration, please contact:
jzhu10@kent.edu
Jianfeng Zhu  
Kent State University
