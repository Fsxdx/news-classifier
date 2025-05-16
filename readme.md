## News Article Topic Classification

A robust NLP pipeline for classifying Russian news articles by topic. This project demonstrates end-to-end
capabilities — from raw data parsing and cleaning through feature engineering, classical baselines, and transformer
fine-tuning in a reproducible, production-ready model.

---

## Project Overview

This repository contains an end-to-end solution for topic classification of news articles in Russian. It covers:

- **Data Collection** news website.
- **Data ingestion** from CSV with `url`, `title`, `text`, `topic`, `tags`, `date`.
- **Cleaning & deduplication**.
- **Exploratory Data Analysis**.
- **Text preprocessing** via spaCy.
- **Classical baselines** (TF-IDF + Logistic Regression, MultinomialNB, LinearSVC, RandomForest, KNN).
- **Transformer fine-tuning** with DeepPavlov’s RuBERT (`DeepPavlov/rubert-base-cased`), including class-weighted loss
  for imbalance.
- **Evaluation** with precision/recall/F1, confusion matrices, macro vs. weighted metrics.
- **Modular Pipeline** definition and serialization.

---

## Preprocessing & EDA

- **Cleaning**
    - Removed duplicate articles
    - Removed articles with similar text and same titles
    - Dropped rows missing `text` or `topic`
    - Created clean versions of texts (lemmatized and without stop words)
- **EDA**
    - Class counts & imbalance analysis
    - Rare classes were removed

---

## Baseline Models

Built with scikit-learn Pipelines (TF-IDF → classifier):

| Model                  | Accuracy | Macro-F1 |
|------------------------|---------:|---------:|
| Logistic Regression    |   0.7343 |   0.6670 |
| MultinomialNB          |   0.7284 |   0.6661 |
| LinearSVC              |   0.6633 |   0.5916 |
| RandomForestClassifier |   0.4503 |   0.3050 |
| KNeighborsClassifier   |   0.1473 |   0.1014 |

Baseline chosen: **Logistic Regression** for best trade-off of speed and performance.

---

## Transformer Fine-Tuning

- **Model**: `DeepPavlov/rubert-base-cased`
- **Loss**: `CrossEntropyLoss` with class weights
- **Hyperparameters**:
    - LR: 2e-5, Batch: 16, Epochs: 3
    - Warmup steps: 500, Weight decay: 0.01

Achieved **Accuracy 0.81**, with macro avg F1 of **0.76**.

---

## Results & Final Analysis

Below is a consolidated comparison of all models on the held-out test set:

| Model                        |   Accuracy |   Macro-F1 |
|------------------------------|-----------:|-----------:|
| Logistic Regression (TF-IDF) |     0.7343 |     0.6670 |
| MultinomialNB (TF-IDF)       |     0.7284 |     0.6661 |
| LinearSVC (TF-IDF)           |     0.6633 |     0.5916 |
| RandomForest (TF-IDF)        |     0.4503 |     0.3050 |
| KNN (TF-IDF)                 |     0.1473 |     0.1014 |
| **Best Baseline (LogReg)**   | **0.7343** | **0.6670** |
| RuBERT Fine-Tuned            |       0.81 |       0.76 |

### Key Takeaways

1. **Transformer vs. Classical**  
   The BERT-based model (RuBERT) outperforms all classical TF-IDF baselines by ~8 pp in accuracy and ~10 pp in macro-F1,
   demonstrating strong capacity to capture contextual semantics.

2. **Class Imbalance Handling**  
   Weighted loss in the transformer training improved recall on minority classes (e.g., “Travels” and “Values”), raising
   their F1-scores by up to 22 pp compared to unweighted runs.

3. **Baseline Strength**  
   Among classical models, Logistic Regression delivered the best speed-performance trade-off.

4. **Error Patterns**
    - **“Business”** remains the hardest to classify consistently, due to vocabulary overlap with
      “Economy” and “Russia.”
    - **High-volume classes** like “Sport” and “Former USSR” achieve > 0.89 F1, showing that ample data yields stable
      performance.

---

## Pipeline & Deployment

- **Data collection code** in `src/collect_data.py`
- **Pipeline code** in `src/pipeline.py`
- **Serialization** via `joblib` (classical) and `model.save_pretrained()` (transformer)  
