# DyME: Empowering Small-scale VLMs with Reliable Thinking Capabilities

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2506.23061-b31b1b.svg)](https://arxiv.org/abs/2506.23061)

This repository provides the official implementation of **DyME** (**Dy**namically selecting between **M**emorization and **E**xploration), accepted at **ICLR 2026**.

## Overview

Small-scale Vision-Language Models (SVLMs) are highly suited for proprietary tasks, but equipping them with reasoning and thinking capabilities remains challenging. Traditional Supervised Fine-Tuning (SFT) can force memorization of pseudo thinking traces, while Reinforcement Learning with Verifiable Reward (RLVR) often leads to unstable exploration (advantage collapse) due to limited model capacity.

**DyME** is a novel training paradigm that dynamically synergizes SFT and RLVR. At each optimization step, DyME dynamically selects between Memorization (via SFT) and Exploration (via RLVR), ensuring every update contributes to an optimal trade-off. To further enhance this, we introduce a **Visual Supervision mechanism** (a visual checker and refiner) to inject dynamically enhanced, image-grounded guidance during training. 

Extensive experiments show that DyME delivers substantial performance improvements, establishing it as a robust strategy for stabilizing SVLM learning.


## Repository Structure

```text
DyME/
├── client_utils/         # Client tools for online Visual Supervision (LLM API)
├── data/                 # Preprocessed textual datasets
├── data_utils/           # Data processing and formatting scripts
│   ├── aokvqa/
│   ├── chart/
│   └── commom_util.py
├── eval/                 # Evaluation scripts for different benchmarks
├── reward_utils/         # Reward function implementations for RLVR
├── config/               # Modular configuration files for experiments
├── default_config.yaml   # Default training and environment configurations
├── main.py               # Entry point for DyME training
├── main_*.py             # Additional experimental variants (e.g., 7B, LLM-only)
├── requirements.txt      # Python dependencies
└── ...
```


## Configuration

Before launching training, please prepare the relevant configuration files. The main settings are managed through configuration files such as `config/config.py` and `default_config.yaml`.

### `CLIENT_CONFIG`

This configuration is required when **Visual Supervision** is enabled. It specifies the online large-model API used by the visual checker and visual refiner.

### `TRAINING_CONFIG`

This section contains standard training hyperparameters for both the memorization phase and the exploration phase, including optimizer settings, batch size, learning rate, and related options.

### `RL_CONFIG`

This section defines critical variables for reward computation and response parsing during RLVR training. In particular, the following delimiters must be properly specified:

* `answer_flag`: used to explicitly separate the final answer from auxiliary generated content such as intermediate reasoning traces.
* `end_flag`: used to mark the end of generation.

These delimiters are essential for stable parsing, reward assignment, and evaluation consistency.



## Data Preparation

We provide example preprocessing scripts in the `data_utils/` directory. After preprocessing, the training data should be organized as a list of dictionaries (e.g., `metadata_list`) following the format below:

```python
metadata_list.append({
    "question": question,               # Full prompt used for training
    "question_wo_prompt": question,     # Raw question without prompt template
    "answer": answer,                   # SFT target; should follow the answer_flag format
    "image": image_save_path,           # Local path to the corresponding image
})
```

### Field Description

* `question`: the complete model input used during training.
* `question_wo_prompt`: the raw question content without any additional prompt wrapper.
* `answer`: the training target for SFT; this field should be formatted consistently with the delimiter specification in `RL_CONFIG`.
* `image`: the local file path of the associated image, if applicable.



## Environment Setup

Please first install the required dependencies and configure the distributed training environment:

```bash
pip install -r requirements.txt
accelerate config
```

The `accelerate config` step is required to initialize the distributed environment for both training and evaluation.



## Dataset Setup

### Text Data

Preprocessed text splits are provided under the `data/` directory.

### Image Data

Due to storage constraints, image datasets are not included in this repository. Please manually place the required image files (e.g., **A-OKVQA** and **ChartQA**) under:

```text
data/images/
```

### Demo Samples

A small subset of demo images for verifying the data loading pipeline may be provided in a future update.



## Dataset Examples

### ChartQA

**ChartQA** is a visual question answering benchmark grounded in chart images. To illustrate different supervision granularities, we provide representative examples with three levels of reasoning-trace quality: **High**, **Medium**, and **Low**.

<div align="center">

| Example                                                         |
| --------------------------------------------------------------- |
| <img src="figs/chartqa.png" alt="ChartQA Example" width="220"/> |

</div>

#### High-quality Example

<details>
<summary><code>High-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "<SUMMARY> To solve the problem, I will examine the image to identify trends in unfavorable views of Pakistan in India over time. I'll closely inspect the data points within the graph to determine the year where the \"very unfavorable view\" reaches its peak. This involves identifying the maximum value on the vertical axis and noting the corresponding year on the horizontal axis. </SUMMARY> \n\n<CAPTION> The image is a line graph titled \"Very unfavorable views of Pakistan increasing in India,\" with the subtitle \"Very unfavorable view of Pakistan.\" The y-axis represents the percentage of unfavorable views, ranging from 0% to 100%. The x-axis displays years from 2013 to 2017. The data points show the percentages of very unfavorable views over these years, with specific values marked: 54% in 2013, 49% in 2014, 51% in 2015, 55% in 2016, and 64% in 2017. The graph shows a general upward trend in unfavorable views, peaking in 2017. </CAPTION> \n\n<REASONING> To determine when the unfavorable view reaches its peak, one should observe the graph for the data point with the highest percentage on the y-axis. The graph shows percentages for each year from 2013 to 2017: starting at 54% in 2013, decreasing to 49% in 2014, and then gradually increasing to 51% in 2015 and 55% in 2016. The graph culminates with the highest percentage of 64% in 2017. Thus, the peak of unfavorable views is associated with the year 2017. </REASONING> \n\n<CONCLUSION> 2017 </CONCLUSION>"
}
```

</details>

#### Medium-quality Example

<details>
<summary><code>Medium-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "Goal: Find the year when the unfavorable view reaches its peak.\nObservation: The data shows the values for each year are: 2013: 0, 2014: 0, 2015: 0, 2016: 55, and 2017: 64.\nReasoning: By comparing the values in each year, the highest value is 64, which occurs in 2017.\nConclusion: The unfavorable view reaches its peak in 2017."
}
```

</details>

#### Low-quality Example

<details>
<summary><code>Low-quality ChartQA Example</code></summary>

```json
{
  "question": "When does the unfavorable view reach the peak?",
  "answer": "2017",
  "hint": "I'm trying to figure out the year when the unfavorable view reaches its highest point. Looking at the data, I see that the values for each year are pretty low until 2016, where it jumps to 55. But the peak doesn't happen until 2017, when the value spikes to 64. So, it seems like the unfavorable view really hits its maximum in 2017."
}
```

</details>


### A-OKVQA

**A-OKVQA** is an open-ended visual question answering benchmark that requires world knowledge, commonsense reasoning, and visual understanding. Below we provide a representative example together with its corresponding annotation.

<div align="center">

| Example                                                        |
| -------------------------------------------------------------- |
| <img src="figs/aokvqa.png" alt="A-OKVQA Example" width="220"/> |

</div>

<details>
<summary><code>View A-OKVQA JSON Example</code></summary>

```json
{
  "question": "What is the man by the bags awaiting?",
  "answer": "cab",
  "visual_fact": "{\n  \"description\": \"The image shows a man standing in the middle of a street, facing away from the camera. He is holding a red bag in one hand and appears to be pulling a black suitcase with wheels. Another black suitcase is lying on the ground near him. The setting is an urban area with houses, parked cars, and trees in the background. The man seems to be waiting or preparing to cross the street.\",\n  \"objects\": [\n    {\n      \"name\": \"man\",\n      \"attributes\": [\"wearing a light blue and white shirt\", \"blue jeans\", \"carrying a red bag\", \"pulling a black suitcase\"],\n      \"position\": \"center\"\n    },\n    {\n      \"name\": \"red bag\",\n      \"attributes\": [\"held by the man\"],\n      \"position\": \"left side of the man\"\n    },\n    {\n      \"name\": \"black suitcase\",\n      \"attributes\": [\"with wheels\", \"being pulled by the man\"],\n      \"position\": \"near the man's feet\"\n    },\n    {\n      \"name\": \"black suitcase\",\n      \"attributes\": [\"on the ground\"],\n      \"position\": \"on the ground near the man\"\n    },\n    {\n      \"name\": \"street\",\n      \"attributes\": [\"asphalt\", \"urban setting\"],\n      \"position\": \"foreground\"\n    },\n    {\n      \"name\": \"houses\",\n      \"attributes\": [\"visible in the background\"],\n      \"position\": \"left side\"\n    },\n    {\n      \"name\": \"parked cars\",\n      \"attributes\": [\"red SUV\", \"other vehicles\"],\n      \"position\": \"left and center background\"\n    },\n    {\n      \"name\": \"trees\",\n      \"attributes\": [\"green foliage\"],\n      \"position\": \"right side\"\n    }\n  ]\n}",
  "hint": "A train would not be on the street, he would not have luggage waiting for a delivery, and the skateboarder is there and not paying attention to him, so a cab is the only plausible answer."
}
```

</details>



### GSM8K

**GSM8K** is a mathematical word problem benchmark. Since it is text-only, we provide a representative question-answer example together with its reasoning trace.

<details>
<summary><code>View GSM8K JSON Example</code></summary>

```json
{
  "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  "answer": "72",
  "hint": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
}
```

</details>



## Training

All training scripts are launched using `accelerate`. The codebase separates the proposed **DyME** framework from standard baseline settings for easier reproduction.

### 1. Training DyME

To train the model with the proposed **DyME** framework, run:

```bash
accelerate launch main.py --config config/config.py
```

### 2. Reproducing Baselines

To reproduce baseline settings such as standard SFT or RL training, use `main_rebuttal.py` and specify the desired mode through `--mode`.

#### Supervised Fine-Tuning (SFT)

```bash
accelerate launch main_rebuttal.py --config config/config.py --mode sft
```

#### Reinforcement Learning (GRPO / RL)

```bash
accelerate launch main_rebuttal.py --config config/config.py --mode grpo
```

### 3. Additional Experimental Variants

For specific experimental settings such as different model scales or architecture-specific ablations, please use the corresponding scripts:

* `main_7B.py`: experiments at the 7B scale
* `main_llm.py`: LLM-specific variants
* `main_change.py`: additional ablation settings


## Evaluation

We support multi-process evaluation through `accelerate`. Evaluation scripts are located in the `eval/` directory and can be launched as Python modules.

### General Usage

```bash
accelerate launch -m eval.<eval_script_name>
```

### Example: ChartQA Evaluation

```bash
accelerate launch -m eval.eval_chartqa
```

### Evaluation Setup

Before running evaluation, please open the corresponding evaluation script (for example, `eval_chartqa.py`) and modify the following fields as needed:

* `model_id`: the path or identifier of the checkpoint to be evaluated
* prompt templates: these should match the formatting used during training

Ensuring consistency between training and evaluation prompts is important for obtaining reliable results.

## Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@inproceedings{dyme2026,
  title={Empowering Small VLMs to Think with Dynamic Memorization and Exploration},
  author={Jiazhen Liu, Yuchuan Deng, Long Chen},
  booktitle={ICLR},
  year={2026},
}
```
