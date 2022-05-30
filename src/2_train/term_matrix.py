# %% [markdown]
# # CountVectorizer

# %%
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

from util import summary_report

# %% [markdown]
# ## Read the data

df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")
df.dropna(subset=["reviewText"], inplace=True)
df.head()

# %% [markdown]
# ## Defining a helpful review + Splitting the data

split = 0.01

df["helpful"] = np.where(df.voteSuccess >= split, 1, 0)

x_train, x_test, y_train, y_test = train_test_split(
    df.reviewText, df.helpful, test_size=0.25, random_state=30
)
f"x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"

# %% [markdown]
# ## Vectorization with CountVectorizer

tm_vectorizer = CountVectorizer(stop_words="english")

term_matrix_train = tm_vectorizer.fit_transform(x_train)
term_matrix_test = tm_vectorizer.transform(x_test)

# %% [markdown]
# ## Fitting

# %% [markdown]
# # Random Forest
rfc = RandomForestClassifier(n_estimators=150, max_depth=150, random_state=0, n_jobs=-1, verbose=True)
rfc.fit(term_matrix_train, y_train)
y_test_pred = rfc.predict(term_matrix_test)

# %%
summary_report(y_test, y_test_pred, "Document-term Matrix(Count Vectorizer) - RandomForestClassifier")

# %% [markdown]
# # LogisticRegression
lgr = LogisticRegression(random_state=0, max_iter=10000)
lgr.fit(term_matrix_train, y_train)
y_test_pred = lgr.predict(term_matrix_test)

# %% 
summary_report(y_test, y_test_pred, "Document-term Matrix(Count Vectorizer) - LogisticRegression")

# %% [markdown]
# # Support-Vector Machine

# %%
from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, max_iter=5000, verbose=True)

clf.fit(term_matrix_train, y_train)
y_test_pred = clf.predict(term_matrix_test)

# %% 
summary_report(y_test, y_test_pred, "Document-term Matrix(Count Vectorizer) - SVM/SVC")

# %% [markdown]
# # Neural Network

# %%
from sklearn.neural_network import MLPClassifier

# Using lbfgs over adam as it is much faster on smaller datasets(1min vs 6min(42k samples on M1 macbook air))
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100,), max_iter=500, random_state=1
)

clf.fit(term_matrix_train, y_train)
y_test_pred = clf.predict(term_matrix_test)

# %% 
summary_report(
    y_test,
    y_test_pred,
    "Document-term Matrix(Count Vectorizer) - Multi-layer Perceptron (MLP)",
)

# %% [markdown]
# # KMeans clustering

# %%
clf = KMeans(n_clusters=2, init="k-means++", max_iter=100, n_init=1, random_state=5)

clf.fit(term_matrix_train, y_train)
y_test_pred = clf.predict(term_matrix_test)

summary_report(y_test, y_test_pred, "MLPClassifier")
# %%
plt.figure(figsize=(10, 8))# Plotting our two-features-space
mtrx_dict = term_matrix_train.todok()
xy = list(mtrx_dict.keys())

colors = ["#FF0000", "#0000FF"]

fig = plt.figure()
ax = fig.add_subplot()

LIMIT = 2500
data = random.sample(list(zip(xy, y_train)), LIMIT)
for i in range(len(data)):
    ax.scatter(x=data[i][0][0], y=data[i][0][1], color=colors[data[i][1]], alpha=0.4)
plt.show()
