# BT5153_Group7_2025
A Machine Learning Approach to Sentiment Analysis on Product Reviews


# BT5153 Project: Sentiment Classification and Clustering of Amazon Reviews

This project is divided into two main parts:
1. **Sentiment Prediction Pipeline** – Builds and compares multiple sentiment classifiers (Logistic Regression, Transformer, Bi-LSTM) and combines them with ensemble learning.
2. **Clustering of Negative Reviews** – Analyzes patterns in negative sentiment reviews using clustering techniques for actionable insight.

---

##  Files

### `sentiment_prediction_pipeline.ipynb`
Contains the full workflow to:
- Preprocess and clean Amazon reviews
- Train and evaluate:
  - Logistic Regression with TF-IDF
  - Transformer (RoBERTa) model via Hugging Face
  - Bi-LSTM with RoBERTa embeddings
- Save predictions as `.json` or `.csv`
- Create a majority-vote ensemble
- Visualize review features (e.g., word clouds)

### `BT5153_Clustering_of_Negative_Reviews.ipynb`
Focuses on:
- Extracting only **negative reviews**
- Preprocessing and embedding generation
- Dimensionality reduction (e.g., PCA, t-SNE)
- Clustering (e.g., KMeans, DBSCAN)
- Visualization of cluster topics for insight generation
  
### `BT5153_Final_LLM.py`
This script uses Google's Gemini 2.0 Flash model to analyze negative customer reviews and generate actionable business insights.
Specifically, the script processes a set of user reviews clustered by common themes and asks the model to:
- Summarize the main negative feedback trends.
- Identify potential root causes for customer dissatisfaction.
- Provide concise recommendations for improvement.
The purpose is to automate customer feedback analysis, enabling businesses to quickly identify issues and prioritize actionable improvements.

---

##  Sentiment Prediction Overview

### Dataset
- Source: [`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- Balanced to 3000 samples per class: `positive`, `neutral`, `negative`

### Model Summary

| Model             | Features                  | Output File                          |
|------------------|---------------------------|--------------------------------------|
| LogisticRegression | TF-IDF (title + text)     | `logistic_predictions.json`          |
| Transformer (RoBERTa) | Fine-tuned Hugging Face model | `transformer_predictions.json`        |
| Bi-LSTM + RoBERTa | RoBERTa embeddings + BiLSTM | `lstm_predictions.json`              |
| Ensemble          | Majority voting            | `majority_voting_predictions.json`   |

Evaluation includes:
- Accuracy
- Macro F1 Score
- Negative Recall
- Classification Report

---

##  Clustering (Negative Reviews)

From `BT5153_Clustering_of_Negative_Reviews.ipynb`:
- Filters negative class reviews
- Applies vectorization (TF-IDF or embeddings)
- Uses unsupervised methods like K-Means or t-SNE
- Visualizes clusters and common themes
- Aims to assist in the root cause analysis of dissatisfaction

---

##  Additional Features

- WordClouds using TF-IDF scores for visual insight
- SHAP/LIME (optional) for explainability
- Misclassification analysis using `review_id`

---

##  Usage

1. Open notebooks in Google Colab
2. Mount Google Drive
3. Run preprocessing once (cached locally)
4. Train models and evaluate results
5. Use the clustering notebook to explore negative sentiment subtopics

---

##  Requirements

- Python ≥ 3.7
- `transformers`, `datasets`, `torch`, `nltk`, `scikit-learn`
- `seaborn`, `matplotlib`, `wordcloud`, `pandas`, `tqdm`

---

##  Author

Group 7  
BT5153, 2025  
Sentiment Analysis & Review Clustering Project
