# %% [markdown]
# # TF-IDF Vectorizer

# %%
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, balanced_accuracy_score

from util import summary_report

# %% [markdown]
# ## Read the data

# %%
df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")

# %% [markdown]
# ## Defining a helpful review + Splitting the data

# %%
split = 0.01

df["helpful"] = np.where(df.voteSuccess >= split, 1, 0)

x_train, x_test, y_train, y_test = train_test_split(
    df.reviewText, df.helpful, test_size=0.25, random_state=30
)
f"x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"

# %% [markdown]
# ## Vectorization with TF-IDF

# %%
vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=0.01)
tfidf_train = vectorizer.fit_transform(x_train.values.astype("U"))
tfidf_test = vectorizer.transform(x_test.values.astype("U"))

# %% [markdown]
# ## Fitting

# %%
clf = LinearSVC(random_state=0, max_iter=10000)

clf.fit(tfidf_train, y_train)
y_test_pred = clf.predict(tfidf_test)

# %% [markdown]
# ## Result

# %%
summary_report(y_test, y_test_pred, "Document-term Matrix(TF-IDF Vectorizer) - SVM/SVC")

# %% [markdown]
# # Neural Network

# %%
from sklearn.neural_network import MLPClassifier

# Using lbfgs over adam as it is much faster on smaller datasets(1min vs 6min(42k samples on M1 macbook air))
clf = MLPClassifier(
    solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100,), max_iter=1000, random_state=1
)

clf.fit(tfidf_train, y_train)
y_test_pred = clf.predict(tfidf_test)

# %%
summary_report(y_test, y_test_pred, "MLPClassifier")
