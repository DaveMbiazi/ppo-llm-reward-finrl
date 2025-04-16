# 🦙 LLaMA PPO Fine-Tuning with Reward Model

This project implements a Reinforcement Learning pipeline using **Proximal Policy Optimization (PPO)** to fine-tune a LLaMA model with a custom reward model. It supports **Supervised Fine-Tuning (SFT)**, **reward model scoring**, and **PPO training** using the **FLARE** dataset.

---

## 🔧 Features

- ✅ **PPO-based fine-tuning of LLaMA models**
- ✅ **Custom value head for critic**
- ✅ **Custom reward function from a fine-tuned classification model**
- ✅ **Support for Supervised Fine-Tuning (SFT)**
- ✅ **Multi-dataset support** (FLARE subsets)
- ✅ **ROUGE-L and reward-based evaluation**
- ✅ **8-bit quantized model loading** (via `bitsandbytes`)

---

## 📚 Dataset

- FLARE (Flexible Language Alignment Reward Evaluation)
  - Supports multiple subsets under the `ChanceFocus` namespace on Hugging Face Datasets.

---

## 🛠️ Components

- **Actor Model**: LLaMA-3-8B with LoRA adapters
- **Critic Model**: LLaMA-3-8B with a custom value head
- **Reward Model**: Fine-tuned classifier evaluating response quality

---

## 🚀 Training Workflow

1. **Supervised Fine-Tuning (optional)** on human-labeled data
2. **Reward Model Scoring** to generate preference scores
3. **PPO Fine-Tuning** using TRL's `PPOTrainer`

---

## 💾 Quantization

- Optional 8-bit quantized loading using `bitsandbytes` for memory efficiency

---

## 📈 Evaluation

- ROUGE-L for lexical similarity
- Custom reward-based scores for alignment quality
