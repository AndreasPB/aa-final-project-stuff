# %%
import json
import gzip
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("averaged_perceptron_tagger")

# %% [markdown]
# # Data Loading
#
# Use `LIMIT` to control the upper limit of objects from each file

# %%
LIMIT = 10000


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if i == LIMIT:
            break
    return pd.DataFrame.from_dict(df, orient="index")


original_reviews = Path("../../data/original")
reviews = [getDF(path) for path in original_reviews.iterdir() if path.is_file()]
df = pd.concat(reviews)
f"Files: {len(reviews)}, Combined lines: {len(df)}"

# %% [markdown]
# # Data Cleaning

# %%
# lowercase review text
df["reviewText"] = df["reviewText"].str.lower()
# remove unverified
df = df[df["verified"] == True]

# removes all rows with empty review texts
df.dropna(subset=["reviewText"], inplace=True)

# remove unixReviewTime
df.drop(
    ["unixReviewTime", "reviewerID", "image", "style", "asin"], axis=1, inplace=True
)


# rename overall to rating
df.rename(columns={"overall": "rating"}, inplace=True)


def cast_to_int(votes):
    try:
        return int(votes)
    except:
        return int(votes.replace(",", ""))


# set votes containing NaN to 0
df["vote"] = df["vote"].fillna(0)
df["vote"] = df["vote"].apply(cast_to_int)
# df["vote"] = df["vote"].astype(int)
# overall to int from float
df["rating"] = df["rating"].astype(int)
# remove NaN reviewText
df["reviewText"] = df["reviewText"].fillna("")

# remove stop words
stop_words = stopwords.words("english")
df["reviewText"] = df["reviewText"].apply(
    lambda x: " ".join([word for word in x.split() if word not in stop_words])
)

# remove punctuation from reviewText
# [^\w\s]' -> looks for anything that isnt a word or whitespace to remove
df["reviewText"] = df["reviewText"].str.replace("[^\w\s]", "")

w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()


# WordNetLemmatizer.lemmatize() only lemmatizes based on tag-parameter given, e.g. "v" for verb, "n" for noun
# This method tries to determine the right tag automatically
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_text(text):
    return [
        lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in w_tokenizer.tokenize(text)
    ]


df["reviewText"] = df["reviewText"].apply(lemmatize_text)
df["reviewText"] = df["reviewText"].apply(lambda x: " ".join(x))

df.to_csv("../../data/cleaned_reviews.tsv", index=False, sep="\t")
