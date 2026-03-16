# DyME: Empowering Small-scale VLMs with Reliable Thinking Capabilities

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23061-b31b1b.svg)](https://arxiv.org/abs/2506.23061)

This repository contains the official implementation of **DyME** (Dynamically selecting between Memorization and Exploration), accepted at **ICLR 2026**.

## 📖 Introduction

Small-scale Vision-Language Models (SVLMs) are highly suited for proprietary tasks, but equipping them with reasoning and thinking capabilities remains challenging. Traditional Supervised Fine-Tuning (SFT) can force memorization of pseudo thinking traces, while Reinforcement Learning with Verifiable Reward (RLVR) often leads to unstable exploration (advantage collapse) due to limited model capacity.

**DyME** is a novel training paradigm that dynamically synergizes SFT and RLVR. At each optimization step, DyME dynamically selects between Memorization (via SFT) and Exploration (via RLVR), ensuring every update contributes to an optimal trade-off. To further enhance this, we introduce a **Visual Supervision mechanism** (a visual checker and refiner) to inject dynamically enhanced, image-grounded guidance during training. 

Extensive experiments show that DyME delivers substantial performance improvements, establishing it as a robust strategy for stabilizing SVLM learning.

## 📁 Repository Structure

```text
DyME/
├── client_utils/         # Client tools for online Vision Supervision (LLM API)
├── data_utils/           # Data processing and formatting scripts
│   ├── aokvqa/
│   ├── chart/
│   └── commom_util.py
├── eval/                 # Evaluation scripts for different benchmarks (e.g., ChartQA)
├── reward_utils/         # Reward function implementations for RLVR
├── config.py             # Global configuration loader
├── default_config.yaml   # Default training and environment configurations
├── DyMETrainer.py        # Core implementation of the DyME training paradigm
├── main.py               # Main entry point for training
└── ...
```

## ⚙️ Configuration

Before training, set up your configurations. The training settings are primarily managed in the configuration files (e.g., `config.py` and `default_config.yaml`).

* **`CLIENT_CONFIG`**: Must be configured when Vision Supervision is required. This sets up the online Large Model API tools used for the visual checker and refiner.
* **`TRAINING_CONFIG`**: Contains standard training hyperparameters for the SFT and RL phases.
* **`RL_CONFIG`**: **Critical variables for reward calculation and response parsing.** You must set the following delimiters:
    * `answer_flag`: Used to strictly separate the pure answer from other text or reasoning traces.
    * `end_flag`: Specifies the termination token/flag.

## 📊 Data Preparation

We provide a demo for data processing in the `data_utils` directory. Your formatted training data should be organized into a list of dictionaries (e.g., `metadata_list`) adhering to this required format:

```python
metadata_list.append({
    "question": question,               # The full prompt used for training
    "question_wo_prompt": question,     # The pure question part (without prompt templates)
    "answer": answer,                   # The SFT label. Must be formatted with the 'answer_flag' linking the pure answer and reasoning steps.
    "image": image_save_path,           # Local file path to the image
})
```

## 🚀 Training

DyME training is based on the Hugging Face `accelerate` environment. 

To launch the training process directly:
```bash
accelerate launch main.py
```
Optional: You can add DeepSpeed configurations to further optimize distributed training across multiple nodes/GPUs.

## 🧪 Evaluation

We support multi-process evaluation. To evaluate your trained model (e.g., on the ChartQA benchmark), use the corresponding script in the eval directory:

```bash
accelerate launch -m eval.eval_chartqa
```

**Important Setup for Evaluation**: Before running the evaluation, open the specific evaluation script (e.g., eval_chartqa.py) and modify:

- The model_id (pointing to your saved checkpoint).
- The prompt templates to ensure they match the formatting used during training.

## 📝 Citation

If you find DyME helpful in your research, please consider citing our paper:

```
@inproceedings{dyme2026,
  title={Empowering Small VLMs to Think with Dynamic Memorization and Exploration},
  author={Jiazhen Liu, Yuchuan Deng, Long Chen},
  booktitle={ICLR},
  year={2026},
}
```