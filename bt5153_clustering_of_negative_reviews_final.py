# -*- coding: utf-8 -*-
"""BT5153_Clustering_of_Negative_Reviews_Final.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1E9dV6-UKPeW3OMmcThqEdRB_XoH6ASDz
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

json_path = '/content/drive/MyDrive/majority_voting_predictions_train.json'
df_test = pd.read_json(json_path, lines=True)

print(df_test.info())
print(df_test.head())

df_reviews = pd.read_csv('/content/drive/MyDrive/final_balanced_reviews.csv')

print(df_reviews.info())
print(df_reviews.head())

df_merged = pd.merge(df_test, df_reviews, on='review_id', how='left')
df_final = df_merged[['review_id', 'text','title', 'majority_pred']]
df_final = df_final[df_final['majority_pred'] == 0]
df = df_final

!pip install kneed

df_final.to_csv("/content/drive/MyDrive/5153_negative_reviews.csv")

"""#TF-IDF + K-means

## Only text TF-IDF + K-means
"""

from sklearn.metrics import silhouette_score
import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator

# 1. Load the data
df = df_final

# 2. Prepare the field (keep only text)
df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.len() > 20].reset_index(drop=True)

# 3. TF-IDF vectorization on text
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english', max_features=5000)
X_text = vectorizer.fit_transform(df["text"])

# 4. Use Elbow Method + Silhouette Score to find optimal K
inertias = []
silhouettes = []
ks = range(2, 15)
print("\n📈 Calculating inertia & silhouette for different K values...")
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_text)
    inertias.append(km.inertia_)
    score = silhouette_score(X_text, labels)
    silhouettes.append(score)
    print(f"K={k}, Inertia={round(km.inertia_, 2)}, Silhouette={round(score, 4)}")

# Plot: dual y-axis for Inertia and Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))

color1 = "tab:blue"
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia", color=color1)
ax1.plot(ks, inertias, marker='o', color=color1)
ax1.tick_params(axis="y", labelcolor=color1)

ax2 = ax1.twinx()
color2 = "tab:green"
ax2.set_ylabel("Silhouette Score", color=color2)
ax2.plot(ks, silhouettes, marker='s', linestyle='--', color=color2)
ax2.tick_params(axis="y", labelcolor=color2)

plt.title("Elbow Method + Silhouette Score (Text Only)")
plt.grid(True)
plt.show()

# Detect elbow point (still use inertia to determine best_k)
kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
best_k = kl.elbow
print(f"\n✅ Optimal K from Elbow: {best_k}")

# 5. Clustering
model = KMeans(n_clusters=best_k, random_state=42)
df["cluster"] = model.fit_predict(X_text)

# 6. Top keywords for each cluster (top 10 terms)
terms = vectorizer.get_feature_names_out()
order_centroids = model.cluster_centers_.argsort()[:, ::-1]

print("\n📌 Top 10 keywords for each cluster (text only):\n")
for i in range(best_k):
    top_words = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_words)}")

# 7. Representative comment for each cluster (closest to centroid)
print("\n📝 Representative comment for each cluster:\n")
closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, X_text)
for i, idx in enumerate(closest):
    print(f"\nCluster {i} Example Review:\n{df.iloc[idx]['text'][:400]}...\n")

# 8. Dimensionality reduction for visualization
svd = TruncatedSVD(n_components=2, random_state=42)
X_svd = svd.fit_transform(X_text)

plt.figure(figsize=(10, 6))
plt.scatter(X_svd[:, 0], X_svd[:, 1], c=df["cluster"], cmap="tab10", s=10)
plt.title(f"KMeans Cluster Visualization (Text Only, K={best_k})")
plt.xlabel("SVD 1")
plt.ylabel("SVD 2")
plt.grid(True)
plt.show()

# 9. Sample 50 texts per cluster (optional)
sampled_df = df.groupby("cluster").apply(lambda x: x.sample(n=min(50, len(x)), random_state=42)).reset_index(drop=True)
sampled_df.to_csv("text_only_cluster_samples.csv", index=False)
print("\n📦 Sampled 50 comments per cluster saved to: text_only_cluster_samples.csv")

"""## encode title and text then combined, TF-IDF + K-means"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score  # ✅ Include silhouette_score
from scipy.sparse import hstack
from kneed import KneeLocator

df = df_final

# Fill missing values and convert to string
df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df = df[df["text"].str.len() > 20].reset_index(drop=True)

# 2. Vectorize title and text separately (preserve frequent and important terms)
vectorizer_title = TfidfVectorizer(max_df=0.8, min_df=3, stop_words='english', max_features=2000)
vectorizer_text = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english', max_features=5000)

X_title = vectorizer_title.fit_transform(df["title"])
X_text = vectorizer_text.fit_transform(df["text"])

# 3. Combine title and text features
X_combined = hstack([X_title, X_text])

# 4. Use Elbow method to find optimal K
inertias = []
silhouette_scores = []  # ✅ Store silhouette scores for each k
ks = range(2, 15)
print("\n📈 Calculating inertia & silhouette for different K values...")
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_combined)
    inertia = km.inertia_
    inertias.append(inertia)

    # ✅ Calculate silhouette score (only if number of samples >= k+1)
    if len(df) > k:
        score = silhouette_score(X_combined, labels)
        silhouette_scores.append(score)
        print(f"K={k}, Inertia={round(inertia, 2)}, Silhouette={round(score, 4)}")
    else:
        silhouette_scores.append(None)
        print(f"K={k}, Inertia={round(inertia, 2)}, Silhouette=N/A")

# Plot: Elbow curve and Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(ks, inertias, marker='o', color='blue', label='Inertia')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("Elbow Method & Silhouette Score")

ax2 = ax1.twinx()
ax2.plot(ks, silhouette_scores, marker='s', color='green', label='Silhouette Score')
ax2.set_ylabel("Silhouette Score", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.grid(True)
plt.show()

# Automatically detect the elbow point
kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
best_k = kl.elbow
print(f"\n✅ Optimal K detected by Elbow: {best_k}")

# 5. Cluster using the optimal K
model = KMeans(n_clusters=best_k, random_state=42)
df["cluster"] = model.fit_predict(X_combined)

# 6. Print top 10 keywords for each cluster (only from text part)
terms = vectorizer_text.get_feature_names_out()
order_centroids = model.cluster_centers_[:, X_title.shape[1]:].argsort()[:, ::-1]

print("\n📌 Top 10 keywords per cluster (based on text only)\n")
for i in range(best_k):
    top_words = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {', '.join(top_words)}")

# 7. Print representative review for each cluster (closest to centroid)
print("\n📝 Representative comment for each cluster:\n")
closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, X_combined)
for i, idx in enumerate(closest):
    print(f"\nCluster {i} Example Review:\n{df.iloc[idx]['title']} | {df.iloc[idx]['text'][:400]}...\n")

# 8. Visualization: TruncatedSVD to 2 dimensions
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_combined)

plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=10)
plt.title(f"KMeans Cluster Visualization (K={best_k})")
plt.xlabel("SVD 1")
plt.ylabel("SVD 2")
plt.grid(True)
plt.show()

"""## title+text combined TF-IDF + K-means"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score  # <- Add silhouette_score if not already included

# ✅ If you haven't installed kneed before, run this once
!pip install -q kneed

from kneed import KneeLocator

# 1. Load all CSVs and concatenate title + text
df = df_final

df["title"] = df["title"].fillna("").astype(str)
df["text"] = df["text"].fillna("").astype(str)
df["text_combined"] = df["title"] + " " + df["text"]
df = df[df["text_combined"].str.len() > 20].reset_index(drop=True)

# 2. TF-IDF vectorization
vectorizer = TfidfVectorizer(max_df=0.9, min_df=5, stop_words='english')
X = vectorizer.fit_transform(df["text_combined"])

# 3. Automatically find the optimal K (Elbow Method + Silhouette Score)
inertias = []
silhouettes = []
ks = list(range(2, 15))

print("\n📈 Calculating Inertia & Silhouette Scores for K from 2 to 14...")
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    inertia = km.inertia_
    silhouette = silhouette_score(X, labels)
    inertias.append(inertia)
    silhouettes.append(silhouette)
    print(f"K = {k}, Inertia = {round(inertia, 2)}, Silhouette = {round(silhouette, 4)}")

# Plot dual-axis chart: Elbow + Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia", color="tab:blue")
ax1.plot(ks, inertias, marker="o", color="tab:blue", label="Inertia")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Silhouette Score", color="tab:green")
ax2.plot(ks, silhouettes, marker="s", linestyle="--", color="tab:green", label="Silhouette Score")
ax2.tick_params(axis="y", labelcolor="tab:green")

plt.title("Elbow Method & Silhouette Score (TF-IDF + Title + Text)")
fig.tight_layout()
plt.grid(True)
plt.show()

# ✅ Detect elbow point (based on Inertia)
kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
best_k = kl.elbow
print(f"\n✅ Optimal K (by Elbow): {best_k}")

"""# BERT + K-means

## text+title combined BERT + KMeans
"""

!pip install -q sentence-transformers

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator  # ✅ 自动找拐点

# 1. Load data
df = df_final
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['combined'] = df['title'] + ' ' + df['text']

# 2. Generate sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)

# 3. Evaluate clustering performance
k_range = range(2, 15)
inertias = []
silhouette_scores = []

print("📈 Calculating inertia & silhouette scores:")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)
    inertia = km.inertia_
    silhouette = silhouette_score(embeddings, labels)
    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    print(f"K={k}, Inertia={round(inertia, 2)}, Silhouette Score={round(silhouette, 4)}")

# 4. Plot Inertia + Silhouette
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(k_range, inertias, marker='o', color='blue', label='Inertia')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("Elbow Method & Silhouette Score")

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, marker='s', color='green', label='Silhouette Score')
ax2.set_ylabel("Silhouette Score", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.grid(True)
plt.show()

# 5. Auto-select best_k using Elbow Method
kl = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
best_k = kl.elbow
print(f"\n✅ Automatically selected K (Elbow Method): {best_k}")

# 6. Final clustering
model_km = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = model_km.fit_predict(embeddings)

# 7. Print representative sample from each cluster
print("\n📌 Representative sample from each cluster (title + first 100 characters of text):\n")
for k in range(best_k):
    sample = df[df['cluster'] == k].iloc[0]
    print(f"Cluster {k}:")
    print(f"Title: {sample['title']}")
    print(f"Text: {sample['text'][:100]}...\n")

# 8. Visualize clustering with PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=10)
plt.title("Sentence-BERT Cluster Visualization")
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(True)
plt.show()

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from kneed import KneeLocator  # 用于自动检测 Elbow 点

# 1. Load the data
df = df_final
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')
df['combined'] = df['title'] + ' ' + df['text']

# 2. Generate sentence embeddings (MPNet: more accurate)
model = SentenceTransformer('all-mpnet-base-v2')
embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)

# 3. KMeans clustering + silhouette & inertia evaluation
k_range = range(2, 15)
silhouette_scores = []
inertias = []

print("📈 Calculating silhouette scores and inertia:")
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    inertia = km.inertia_
    silhouette_scores.append(score)
    inertias.append(inertia)
    print(f"K={k}, Inertia={round(inertia, 2)}, Silhouette Score={round(score, 4)}")

# 4. Visualize Inertia + Silhouette Score
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(k_range, inertias, marker='o', color='blue', label='Inertia')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title("Elbow Method & Silhouette Score")

ax2 = ax1.twinx()
ax2.plot(k_range, silhouette_scores, marker='s', color='green', linestyle='--', label='Silhouette Score')
ax2.set_ylabel("Silhouette Score", color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.grid(True)
plt.show()

# 5. Automatically select best K using Elbow Method
kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
best_k = kl.elbow
print(f"\n✅ Automatically selected best_k (elbow): {best_k}")

# 6. Final KMeans clustering
model_km = KMeans(n_clusters=best_k, random_state=42)
df['cluster'] = model_km.fit_predict(embeddings)

# 7. Print representative sample from each cluster
print("\n📌 Representative samples from each cluster (title + first 100 characters of text):\n")
for k in range(best_k):
    sample = df[df['cluster'] == k].iloc[0]
    print(f"Cluster {k}:")
    print(f"Title: {sample['title']}")
    print(f"Text: {sample['text'][:100]}...\n")

# 8. Dimensionality reduction visualization using PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=df["cluster"], cmap="tab10", s=10)
plt.title("Sentence-BERT Cluster Visualization (K={})".format(best_k))
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.grid(True)
plt.show()

"""# BERTopic + Mapnet + Reduced"""

# Install dependencies (run only once)
!pip install bertopic
!pip install umap-learn

# Step 1: Import libraries
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# Step 2: Load data and concatenate title + text
df = pd.read_csv("/content/drive/MyDrive/5153_negative_reviews.csv")
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")
df["combined"] = df["title"] + " " + df["text"]

# Step 3: Generate sentence embeddings using a strong model (MPNet)
embedding_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedding_model.encode(df["combined"], show_progress_bar=True)

# Step 4: Define improved vectorizer (remove stopwords, allow n-grams)
vectorizer_model = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5
)

# Step 5: Perform BERTopic clustering with custom vectorizer
topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    verbose=True
)
topics, probs = topic_model.fit_transform(df["combined"], embeddings)

# Step 6: Reduce topic count to top 25 topics
topic_model = topic_model.reduce_topics(df["combined"], nr_topics=25)

# Re-transform to update topic assignments
topics_reduced, probs_reduced = topic_model.transform(df["combined"])
df["topic"] = topics_reduced
df["topic_prob"] = probs_reduced

# Step 7: Export selected fields
cols_to_export = ["review_id", "title", "text", "topic", "topic_prob"]
df_export = df[cols_to_export]
df_export.to_csv("bertopic_clustered_reviews_filtered.csv", index=False)

# Optional: Download CSV
from google.colab import files
files.download("bertopic_clustered_reviews_filtered.csv")

# Step 8: Display topic summary
print("📌 Top Topics Summary (After Reduction):")
topic_info = topic_model.get_topic_info()
print(topic_info.head(10))

# Step 9: Print keywords and a representative comment for each topic
print("\n📝 Representative keywords and comments for each topic:\n")

for topic_num in topic_info["Topic"].tolist():
    if topic_num == -1:
        continue  # Skip noise topic
    # Get top keywords
    top_words = topic_model.get_topic(topic_num)[:5]
    keyword_list = [w[0] for w in top_words]

    # Get representative comment (highest topic_prob)
    top_idx = df[df["topic"] == topic_num]["topic_prob"].idxmax()
    top_text = df.loc[top_idx, "combined"][:200].replace("\n", " ").strip()

    print(f"🔹 Topic {topic_num}: {', '.join(keyword_list)}")
    print(f"   → {top_text}\n")

# Step 10: Visualize topics (interactive, works best in Colab)
topic_model.visualize_topics()

# Step 11: Export original fields along with reduced topic results
cols_to_export = ["review_id", "title", "text", "topic"]
df_export = df[cols_to_export]

# Save as CSV file
df_export.to_csv("/content/drive/MyDrive/bertopic_clustered_reviews_filtered.csv", index=False)