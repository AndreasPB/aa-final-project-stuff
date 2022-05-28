# %%
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %%
EXPORT_PLOTS = False
df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")

# %%
df

# %%
five_star_reviews = df.loc[df.rating == 5]
one_star_reviews = df.loc[df.rating == 1]

# %%
len(five_star_reviews)

# %%
len(one_star_reviews)

# %%
one_star_words = one_star_reviews.assign(
    word=one_star_reviews["reviewText"].str.split()
).explode("word")["word"]
five_star_words = five_star_reviews.assign(
    word=five_star_reviews["reviewText"].str.split()
).explode("word")["word"]
one_star_words.head()

# %%
counted_one_star: dict[str, int] = (one_star_words.value_counts()).head(1000).to_dict()
counted_five_star: dict[str, int] = (
    (five_star_words.value_counts()).head(1000).to_dict()
)

# %%
wc_one_star = WordCloud(
    width=1200, height=500, max_words=500
).generate_from_frequencies(counted_one_star)
wc_five_star = WordCloud(
    width=1200, height=500, max_words=500
).generate_from_frequencies(counted_five_star)
if EXPORT_PLOTS:
    wc_one_star.to_file("../../report/img/one_star_wordcloud.png")
    wc_five_star.to_file("../../report/img/five_star_wordcloud.png")

# %%
plt.figure(figsize=(10, 10))
plt.imshow(wc_one_star)
plt.axis("off")
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.imshow(wc_five_star)
plt.axis("off")
plt.show()
