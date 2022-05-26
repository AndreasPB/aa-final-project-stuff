# %%
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# %%
df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")
df.dropna(subset=["reviewText"], inplace=True)
df.head()

# %%
# could use LogisticRegression classifier to map our numbers in the range [0,1]
# check accuracy and also youden's index
    # - use youden's index to get indication of our predictive power
# Compare how it performes with TF-IDF

split = 0.1

df["helpful"] = np.where(df.voteSuccess >= split, 1, 0)

x_train, x_test, y_train, y_test = train_test_split(df.reviewText, df.helpful, test_size=0.25, random_state=30)

# %%
tm_vectorizer = CountVectorizer(stop_words="english")

term_matrix_train = tm_vectorizer.fit_transform(x_train)
term_matrix_test = tm_vectorizer.transform(x_test)

# %%
clf = LinearSVC(random_state=0, max_iter=10000)

clf.fit(term_matrix_train, y_train)
y_test_pred = clf.predict(term_matrix_test)

# %%
print("Document-term Matrix(Count Vectorizer) - SVM/SVC")
print(classification_report(y_test, y_test_pred, target_names=["Unhelpful", "Helpful"]))

# %%
plt.figure(figsize=(10, 8))# Plotting our two-features-space
mtrx_dict = term_matrix_train.todok()
xy = list(mtrx_dict.keys())

colors=["#FF0000", "#0000FF"]

fig = plt.figure()
ax = fig.add_subplot()

LIMIT = 2500
data = random.sample(list(zip(xy, y_train)), LIMIT)
for i in range(len(data)):
    ax.scatter(x=data[i][0][0], y=data[i][0][1], color=colors[data[i][1]], alpha=0.4)
plt.show()
