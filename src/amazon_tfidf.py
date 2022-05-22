# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk


# %%
df = pd.read_csv("data/cleaned_reviews.tsv", sep="\t")

df["rating"] = df["rating"].astype(int)
df = df[df["rating"] != 3]
df["label"] = np.where(df["rating"] >= 4, 1, 0)

# Split into positive and negative reviews
df_positive = df.loc[df["label"] == 1]
df_negative = df.loc[df["label"] == 0]

# Vectorization, wooo
vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=0.01)
positive_features = vectorizer.fit_transform(df_positive["reviewText"].values.astype('U'))
negative_features = vectorizer.fit_transform(df_negative["reviewText"].values.astype('U'))
print(positive_features)
