# %%
import pandas as pd
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
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

# rename overall to rating
df.rename(columns = {"overall": "rating"}, inplace=True)

# set votes containing NaN to 0
df["vote"] = df["vote"].fillna(0)
# overall to int from float
df["rating"] = df["rating"].astype(int)
# remove NaN reviewText
df["reviewText"] = df["reviewText"].fillna("")

# remove stop words
stop_words = stopwords.words("english")
df["reviewText"] = df["reviewText"].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# remove punctuation from reviewText
# [^\w\s]' -> looks for anything that isnt a word or whitespace to remove
df["reviewText"] = df["reviewText"].str.replace('[^\w\s]',"")

w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()


# WordNetLemmatizer.lemmatize() only lemmatizes based on tag-parameter given, e.g. "v" for verb, "n" for noun
# This method tries to determine the right tag automatically
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)]


df["reviewText"] = df["reviewText"].apply(lemmatize_text)
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x))

df.to_csv("magazine_reviews_cleanup.tsv", index=False, sep="\t")
