# LLM-Personality-Inference-Interviews
Personality inference from semi-structured interviews using LLMs (prompting, LoRA, fine-tuning, and embedding-based regression).
📄 Title

Personality Inference from Semi-Structured Interviews: A Comparative Evaluation of Large Language Models

🔍 Overview

This repository contains code and resources for our study on personality trait prediction using large language models (LLMs) from semi-structured interview transcripts.

We systematically compare four modeling paradigms:

Prompt-based inference (GPT-5 Mini, zero-shot & CoT)
Parameter-efficient fine-tuning (LoRA; RoBERTa & LLaMA)
Supervised fine-tuning (GPT-4.1 Mini)
Embedding-based regression (MPNet, E5, OpenAI embeddings)
📊 Key Findings
Prompt-based methods show weak and unstable performance
Supervised fine-tuning achieves strong correlations (r ≈ 0.72)
Embedding-based regression provides competitive performance
Trait prediction is highly trait-dependent
Conscientiousness & Agreeableness → strong
Openness → consistently difficult
📂 Dataset

We use a real-world dataset of:

N = 518 participants
Semi-structured interview transcripts (~15 minutes each)
Ground-truth Big Five scores (BFI-10)

🔗 Dataset available via OSF:
👉 https://osf.io/357yk/overview?view_only=12566422d5394a6eb78b89e6e597fe01

🧠 Prompt Design
Zero-shot

Direct trait prediction from interview text

Chain-of-Thought (CoT)

Step-by-step reasoning:

Extract behavioral evidence
Reason per trait
Assign scores

(See /prompts/ for full templates)

⚙️ Methods
1. Prompt-based Inference
GPT-5 Mini
Default inference settings
Deterministic output formatting
2. LoRA Fine-tuning
Models: RoBERTa-base, LLaMA-2-7B
Rank: 8 / 16 / 32
Chunk size: 512 tokens
3. Supervised Fine-tuning
Model: GPT-4.1 Mini
80/20 split
3 epochs
4. Embedding-based Regression
Encoders:
all-mpnet-base-v2
E5-large-v2
text-embedding-3-small
text-embedding-3-large
Model: Multi-output Ridge regression
📈 Evaluation
Pearson correlation (r)
Mean Absolute Error (MAE)
Statistical significance (p-values, CI)
⚠️ Limitations
BFI-10 reliability ceiling (α ≈ 0.65–0.75)
Long-text chunking may dilute signals
Sample size (N=518)
🔮 Future Work
Multimodal personality inference (video, facial cues)
Hierarchical encoding for long conversations
Domain-adapted LLMs
