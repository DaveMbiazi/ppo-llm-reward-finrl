# Flan-T5 PPO Fine-Tuning with Custom Reward Model

This project implements a Reinforcement Learning pipeline using **Proximal Policy Optimization (PPO)** to fine-tune a **T5-based actor model** (e.g., Flan-T5) with a **custom reward model** and **critic network**. It includes **Supervised Fine-Tuning (SFT)**, **reward-based PPO updates**, and **evaluation** across various **FLARE datasets**.

---

## 🔧 Features

- ✅ PPO-based fine-tuning of T5-based actor models
- ✅ Custom value head for critic (Flan-T5 encoder)
- ✅ Reward function using a fine-tuned language model
- ✅ Supervised Fine-Tuning (SFT) stage prior to PPO
- ✅ Dataset support for multiple FLARE variants from Hugging Face (`ChanceFocus` namespace)
- ✅ ROUGE-L and reward-based evaluation metrics
- ✅ Optional 8-bit quantization support with `bitsandbytes`

---

## 🛠️ Components

- **Actor Model**: `google/flan-t5-base` (or similar), trained with SFT and PPO
- **Critic Model**: Flan-T5 encoder with added value head
- **Reward Model**: Pretrained language model evaluating alignment quality
- **Datasets**: From the [ChanceFocus](https://huggingface.co/datasets/ChanceFocus) namespace on Hugging Face

---

## 📚 Supported Datasets

- `flare-german`
- `flare-fiqasa`
- `flare-es-instruction-tuning`
- `flare-es-stock`
- `en-fpb`
- `m2sum`
- `pubmedsum`
- `flare-headlines`
- `flare-finqa`
- `flare-convfinqa`
- `flare-ner`
- `flare-sm-cikm`
- `flare-sm-acl`

---

## ⚙️ Training Pipeline

1. **Supervised Fine-Tuning** on labeled data
2. **Reward Model Scoring** for PPO feedback
3. **PPO Fine-Tuning** using advantage estimation and clipped objective
4. **Evaluation** using ROUGE-L and learned reward function

---

## 🧪 Evaluation Metrics

- **ROUGE-L** for comparing generation and reference
- **Custom reward scores** from the trained reward model
- **Critic value estimates**

---

## 🗃️ Output

- Fine-tuned actor model saved for each dataset
- Evaluation logs and CSVs:
  - `test_results_<dataset>.csv`
  - `val_results_<dataset>.csv`
- Final model saved at: `final_actor_model/`

---

## 💡 Notes

- PPO uses `CrossEntropyLoss` for policy updates
- Reward model is assumed to be stored at `reward_model_final`
- Adjust `MAX_LENGTH`, `LEARNING_RATE`, or `BATCH_SIZE` in the config as needed
- Enable 8-bit quantization by uncommenting and configuring `BitsAndBytesConfig`

---

## 🚀 Run

Ensure dependencies are installed:

```bash
pip install -r requirements.txt

