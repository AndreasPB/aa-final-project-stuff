# %%
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# %%
df = pd.read_csv("../data/cleaned_reviews.tsv", sep="\t")

# %%
vectorizer = CountVectorizer(stop_words="english")
df.dropna(subset=["reviewText"], inplace=True)
data_vec = vectorizer.fit_transform(df.reviewText)

term_matrix = pd.DataFrame(data=data_vec.toarray(), columns=vectorizer.get_feature_names_out())
term_matrix

# %%
