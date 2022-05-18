# %%
import pandas as pd
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
import json
import gzip

# %%
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('Magazine_Subscriptions.json.gz')


# %%

# Data cleanup

# lowercase review text
df["reviewText"] = df["reviewText"].str.lower()
# remove unverified
df = df[df["verified"] == True]
# remove unixReviewTime
df.drop(["unixReviewTime","reviewerID", "image", "style", "asin"], axis=1, inplace=True)

# set votes containing NaN to 0
df["vote"] = df["vote"].fillna(0)
# overall to int from float
df["overall"] = df["overall"].astype(int)
# remove NaN reviewText
df["reviewText"] = df["reviewText"].fillna("")

# remove stop words
tokenizer = ToktokTokenizer()
stop_words = stopwords.words("english")
df["reviewText"] = df["reviewText"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# remove punctuation from reviewText
# [^\w\s]' -> looks for anything that isnt a word or whitespace to remove
df["reviewText"] = df["reviewText"].str.replace('[^\w\s]',"")

# MISSING:
# - stemming
# - lemmatization

df.to_excel("magazine_reviews_cleanup.xlsx", index=False)


