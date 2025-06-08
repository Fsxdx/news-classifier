## News Article Topic Classification

NLP pipeline for classifying Russian news articles by topic. This project demonstrates end-to-end
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
- **Evaluation** with precision/recall/F1.
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
| Logistic Regression    | 0.755383 | 0.732552 |
| MultinomialNB          | 0.692399 | 0.730963 |
| LinearSVC              | 0.751107 | 0.654706 |
| RandomForestClassifier | 0.450096 | 0.326722 |
| KNeighborsClassifier   | 0.152505 | 0.115004 |

Baseline chosen: **Logistic Regression** for best trade-off of speed and performance.

---

## Transformer Fine-Tuning

- **Model**: `DeepPavlov/rubert-base-cased`
- **Loss**: `CrossEntropyLoss` with class weights
- **Hyperparameters**:
    - LR: 1e-5, Batch: 128, Epochs: 4
    - Warmup steps: 0.05, Weight decay: 0.01

Achieved **Accuracy 0.80**, with macro avg F1 of **0.80**.

---

## Results & Final Analysis

| Metric      | TF-IDF + LogReg | RuBERT   | Δ         |
|-------------|-----------------|----------|-----------|
| Accuracy    | **0.75**        | **0.80** | **+0.05** |
| Macro F1    | 0.73            | 0.80     | **+0.07** |
| Weighted F1 | 0.75            | 0.82     | **+0.07** |

*Interpretation*: A 5 pp accuracy gain on a 39 k-article test set is statistically very strong. Macro-F1 improves even
more, confirming that rare classes benefit most.

---

### Class-level insights

| Label (support)       | Biggest change                                 | Comments                                                                                                 |
|-----------------------|------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Путешествия**       | Recall 0.42 → 0.86, <br/>F1 0.58 → 0.80        | Contextual clues (“flight”, “visa”, country names) are captured by BERT but lost in sparse TF-IDF space. |
| **Силовые структуры** | Recall 0.40 → 0.86, <br/>F1 0.44 → 0.59        | BERT recognizes entity patterns (“FSB”, “ МВД ”) but precision remains low.                              |
| **Дом**               | Precision 0.75 → 0.54, <br/>F1 0.78 → 0.67     | Small class, many lifestyle words overlap with *Values* and *Life*.                                      |
| **Мир**               | Precision 0.80 → 0.93, <br/>Recall 0.79 → 0.76 | BERT is stricter: fewer false positives, more false negatives.                                           |
| **Спорт**             | Near-perfect both models (0.96 vs 0.98)        | High lexical uniqueness; baseline already saturated.                                                     |


---

### Practical implications

| Aspect                        | TF-IDF + LogReg | RuBERT                  |
|-------------------------------|-----------------|-------------------------|
| Training time                 | Minutes on CPU  | \~2.5 h on single GPU   |
| Inference latency (batch = 1) | 2 ms            | 25 - 35 ms              |
| Memory                        | 200 MB          | 425 MB                  |
| Maintenance                   | Simple re-train | Need GPU & HF ecosystem |

## Pipeline & Deployment

- **Data collection code** in `src/data/collect_data.py`
- **Pipeline code** in `src/data/pipeline.py`