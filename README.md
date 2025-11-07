#  Team Project 2 (Team11): 
# A Study on LLM Response Preference Prediction and Classification
## CS 53744 Machine Learning Project - Instructor: Professor Jongmin Lee

**Team Members:** 20222446 Hyounsoo Kim, 20201876 Sanghyun Na, 20203009 Jaehyun Park

## Folder Structure
```markdown
root
├── step1/
│   ├── main.ipynb
│   └── submission.csv
├── step2/
│   ├── main.ipynb
│   └── submission.csv
├── step3-4/
│   ├── main.ipynb
│   └── submission.csv
└── step5/
    ├── main.ipynb
    ├── kaggle.ipynb # to submission kaggle for inference
    └── submission.csv
```



**Project Goal:** To develop a multi-class classification model that predicts the human-annotated winner (A Wins, B Wins, or Tie) between two competing LLM responses (A, B) for a given prompt, using the Kaggle 'LLM Classification Finetuning' dataset.

**Evaluation Metric:** Multi-class Log Loss (Target: Minimization)

---

##  Abstract

This project tackles the complex classification task of predicting human preference between two competing Large Language Model (LLM) responses. We implemented several machine learning strategies, progressing from simple statistical baselines to a complex neural architecture.

**Final Model Architecture:** A **DeBERTa-v3-small Dual Encoder** fine-tuned with **LoRA**, critically incorporating a **Cross-Attention** mechanism to explicitly model the relational context between the two responses. The final classifier is augmented with difference and product feature vectors.

**Key Achievements:**

* **Lowest Validation Log Loss:** **1.01768**.
* **Core Finding:** The model exhibited a significant **Verbosity Bias** (favoring longer, detailed responses) and showed the greatest difficulty in predicting the ambiguous **'Tie'** class (Log Loss 1.13130). This suggests the need for specialized ranking loss functions in future work.
* **LLaMA 3 8B Attempt:** An approach utilizing a LLaMA 3 Dual Encoder was explored but did not exceed the performance of our efficient DeBERTa-based state-of-the-art model.

---

## Reproducing Our State-of-the-Art LLM Preference Predictor

This guide outlines the steps to reproduce the results of our final, state-of-the-art model: the **LoRA-tuned DeBERTa-v3-small Dual Encoder with Cross-Attention**.

### 1. Environment Setup & Prerequisites

All steps rely on fixed configurations to ensure complete reproducibility.

* **Environment:** A GPU environment (e.g., Google Colab, a server with a T4 or equivalent GPU) is required.
* **Python Version:** 3.x (Project used 3.13.5)
* **Random Seed:** **`random_state=42`** is fixed for all data splits and model initialization.
* **Dependency Installation:**
    ```bash
    pip install torch transformers datasets accelerate bitsandbytes peft scikit-learn pandas numpy
    ```

### 2. Data Preparation 

1.  **Data Download:** Download the `train.csv` file from the Kaggle 'LLM Classification Finetuning' competition and save it in the `data/` directory.
2.  **Data Splitting:** To replicate our reported Validation Log Loss, you must split the training data:
    * Load `train.csv`.
    * Perform a **stratified split** on the target `winner` column (A, B, Tie) using a percentage (e.g., 10%) and set **`random_state=42`**.

### 3. Final Model Architecture and Configuration 

| Component | Detail | Note |
| :--- | :--- | :--- |
| **Backbone Model** | `microsoft/deberta-v3-small` | Used as a Shared-Weight Dual Encoder |
| **Fine-tuning Technique** | **LoRA (Low-Rank Adaptation)** | Used for efficient training |
| **LoRA Target Layers** | `query_proj`, `value_proj` | Attention Projection Layers |
| **LoRA Hyperparameters** | `r=16`, `lora_alpha=32`, `lora_dropout=0.05` | Aggressive rank setting; only ~0.2% of parameters are trained |
| **Core Innovation** | **Cross-Attention Layer** | Explicitly models the comparison between two response embeddings |
| **Classification Features** | Concatenation, **Difference**($v_A - v_B$), **Product**($v_A \odot v_B$) | Augments the final classification layer |

### 4. Model Training and Execution 

Use the following parameters in your training script (e.g., `scripts/train_final_model.py`):

| Training Parameter | Value |
| :--- | :--- |
| **Model** | DeBERTa-v3-small + LoRA |
| **Max Sequence Length** | 512 |
| **Batch Size** | 8 or 16 | (Adjust based on GPU memory) |
| **Learning Rate (LR)** | $1 \times 10^{-4}$ |
| **Optimizer** | AdamW |
| **Epochs** | 3-5 | (Monitor Validation Loss for early stopping) |
| **Estimated Training Time** | Approx. 1.5 hours (on T4 GPU) |

**Example Execution Command:**

```bash
# Train the DeBERTa-v3-small Cross-Attention model with LoRA
python scripts/train_final_model.py --model_name "deberta-v3-small" --use_cross_attention True --lora_rank 16 --random_seed 42
