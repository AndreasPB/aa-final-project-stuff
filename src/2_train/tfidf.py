# %% [markdown]
# # TF-IDF Vectorizer

# %%
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# %% [markdown]
# ## Read the data

# %%
df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")

# %% [markdown]
# ## Defining a helpful review + Splitting the data

# %%
split = 0.1

df["helpful"] = np.where(df.voteSuccess >= split, 1, 0)

x_train, x_test, y_train, y_test = train_test_split(df.reviewText, df.helpful, test_size=0.25, random_state=30)

# %% [markdown]
# ## Vectorization with TF-IDF

# %%

vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=0.01)
tfidf_train = vectorizer.fit_transform(x_train.values.astype('U'))
tfidf_test = vectorizer.transform(x_test.values.astype('U'))

# %% [markdown]
# ## Fitting

# %%
clf = LinearSVC(random_state=0, max_iter=10000)

clf.fit(tfidf_train, y_train)
y_test_pred = clf.predict(tfidf_test)

# %% [markdown]
# ## Result

# %%
print("Document-term Matrix(Count Vectorizer) - SVM/SVC")
print(classification_report(y_test, y_test_pred, target_names=["Unhelpful", "Helpful"]))
