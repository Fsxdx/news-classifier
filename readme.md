## News Article Topic Classification

A robust NLP pipeline for classifying Russian news articles by topic. This project demonstrates end-to-end capabilities—from raw data parsing and cleaning through feature engineering, classical baselines, and transformer fine-tuning—culminating in a reproducible, production-ready model.

---

## Project Overview

This repository contains an end-to-end solution for topic classification of news articles in Russian. It covers:

- **Data ingestion** from CSV with `url`, `title`, `text`, `topic`, `tags`, `date`.  
- **Cleaning & deduplication**.
- **Exploratory Data Analysis**.  
- **Text preprocessing** via spaCy.  
- **Classical baselines** (TF-IDF + Logistic Regression, MultinomialNB, LinearSVC, RandomForest, KNN).  
- **Transformer fine-tuning** with DeepPavlov’s RuBERT (`DeepPavlov/rubert-base-cased`), including class-weighted loss for imbalance.  
- **Evaluation** with precision/recall/F1, confusion matrices, macro vs. weighted metrics.  
- **Modular Pipeline** definition and serialization.  

---

## Preprocessing & EDA

- **Cleaning**  
  - Removed duplicate articles  
  - Dropped rows missing `text` or `topic`  
- **EDA**  
  - Class counts & imbalance analysis  
  - Top-20 tag distribution  

---

## Baseline Models

Built with scikit-learn Pipelines (TF-IDF → classifier):

| Model                     | Accuracy | Macro-F1 |
|---------------------------|---------:|---------:|
| Logistic Regression       |   0.7581 |   0.7333 |
| MultinomialNB             |   0.7425 |   0.7120 |
| LinearSVC                 |   0.7634 |   0.7398 |
| RandomForestClassifier    |   0.7458 |   0.7182 |
| KNeighborsClassifier      |   0.7012 |   0.6655 |

Baseline chosen: **LinearSVC** for best trade-off of speed and performance.

---

## Transformer Fine-Tuning

- **Model**: `DeepPavlov/rubert-base-cased`  
- **Loss**: `CrossEntropyLoss` with class weights  
- **Train/Val/Test split** stratified by `topic`  
- **Hyperparameters**:  
  - LR: 2e-5, Batch: 16, Epochs: 3  
  - Warmup steps: 500, Weight decay: 0.01  

Achieved **Accuracy 0.8229**, with macro avg F1 of **0.80**.

---

## Results

**Class-level metrics (Transformer)**  

| Topic               | Precision | Recall | F1-Score | Support |
|---------------------|----------:|-------:|---------:|--------:|
| Business            |      0.52 |   0.69 |     0.59 |     926 |
| Former USSR         |      0.95 |   0.89 |     0.92 |    2471 |
| Home                |      0.88 |   0.49 |     0.63 |    1453 |
| …                   |       …   |    …   |      …   |     …   |
| Economy             |      0.87 |   0.69 |     0.77 |    2552 |
| **Overall**         |           |        |          |         |
| Accuracy            |       —   |    —   |   0.8229 |   36761 |
| Macro avg F1-score  |       —   |    —   |     0.80 |         |
| Weighted avg F1-score |     —   |    —   |     0.82 |         |

---

## Pipeline & Deployment

- **Pipeline code** in `src/pipeline.py`  
- **Serialization** via `joblib` (classical) and `model.save_pretrained()` (transformer)  
