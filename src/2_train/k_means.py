# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# %%
# read data into a DataFrame
data = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")

#make a copy of columns I need from raw data
df1 = data.iloc[:, [4,5,6,9]]
df1.head()

# %%
df1["helpful"] = np.where(data.voteSuccess >= 0.01, 1, 0)

df1.head(100)

# %%
#make a copy
df2 = df1.copy(deep = True)

# %%
#tokenize text with Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 0.1, max_df=0.9,
                             ngram_range=(1, 4), 
                             stop_words='english')
vectorizer.fit(df2['reviewText'])

# %%
#tokenize text with Tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df = 0.1, max_df=0.9,
                             ngram_range=(1, 4), 
                             stop_words='english')
vectorizer.fit(df2['reviewText'])

# %%
X_train = vectorizer.transform(df2['reviewText'])
vocab = vectorizer.get_feature_names()

# %%
#find best logistic regression parameters
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
feature_set = X_train
gs = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid={'C': [10**-i for i in range(-5, 5)], 'class_weight': [None, 'balanced']},
    scoring='roc_auc'
)

gs.fit(X_train, df2.helpful)

# %%
#plot ROC/AUC curve
from sklearn.metrics import roc_auc_score, roc_curve
actuals = gs.predict(feature_set) 
probas = gs.predict_proba(feature_set)
plt.plot(roc_curve(df2[['helpful']], probas[:,1])[0], roc_curve(df2[['helpful']], probas[:,1])[1])

# %%
# ROC/AUC score
y_score = probas
test2 = np.array(list(df2.helpful))
y_true = test2
roc_auc_score(y_true, y_score[:,1].T)

# %%
#Apply TfidfVectorizer to review text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

model = KMeans(n_clusters=4, init='k-means++', max_iter=100, n_init=1,random_state=5)

vectorizer = TfidfVectorizer(min_df = 0.05, max_df=0.95,
                             ngram_range=(1, 2), 
                             stop_words='english')
vectorizer.fit(df1['reviewText'])

# %%
X_train = vectorizer.transform(df1['reviewText'])
vocab = vectorizer.get_feature_names()
sse_err = []
res = model.fit(X_train)
vocab = np.array(vocab)
cluster_centers = np.array(res.cluster_centers_)
sorted_vals = [res.cluster_centers_[i].argsort() for i in range(0,np.shape(res.cluster_centers_)[0])]
words=set()
for i in range(len(res.cluster_centers_)):
    words = words.union(set(vocab[sorted_vals[i][-10:]]))
words=list(words)

print(words)

# %%
#add top words to train set
train_set=X_train[:,[np.argwhere(vocab==i)[0][0] for i in words]]

# %%
# how many observations are in each cluster
df1['cluster'] = model.labels_
df1.groupby('cluster').count()

# %%
# what does each cluster look like
df1.groupby('cluster').mean()

# %%
# correlation matrix
df1.corr()


# %%
#add top words to train set
train_set=X_train[:,[np.argwhere(vocab==i)[0][0] for i in words]]


# %%
# how many observations are in each cluster
df1['cluster'] = model.labels_
df1.groupby('cluster').count()

# %%
# what does each cluster look like
df1.groupby('cluster').mean()

# %%
# correlation matrix
df1.corr()


