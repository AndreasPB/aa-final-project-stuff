# %%
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("magazine_reviews_cleanup.tsv", sep="\t")

# %%
df

# %%
five_star_reviews = df.loc[df.overall == 5]
one_star_reviews = df.loc[df.overall == 1]

# %%
len(five_star_reviews)

# %%
len(one_star_reviews)

# %%
one_star_words = one_star_reviews.assign(word = one_star_reviews["reviewText"].str.split()).explode("word")["word"]
five_star_words = five_star_reviews.assign(word = five_star_reviews["reviewText"].str.split()).explode("word")["word"]
one_star_words.head()

# %%
counted_one_star: dict[str, int] = (one_star_words.value_counts()).head(1000).to_dict()
counted_five_star: dict[str, int] = (five_star_words.value_counts()).head(1000).to_dict()

# %%
wc_one_star = WordCloud(width=1200, height=500, max_words=500).generate_from_frequencies(counted_one_star)
wc_one_star.to_file("one_star_wordcloud.png")
wc_five_star = WordCloud(width=1200, height=500, max_words=500).generate_from_frequencies(counted_five_star)
wc_five_star.to_file("five_star_wordcloud.png")

# %%
plt.figure(figsize=(10, 10))
plt.imshow(wc_one_star)
plt.axis('off')
plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.imshow(wc_five_star)
plt.axis('off')
plt.show()
