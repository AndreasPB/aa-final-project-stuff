# %%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("magazine_reviews_cleanup.tsv", sep="\t")

# %%
ax = data['rating'].value_counts().sort_index().plot(kind='bar', rot=0)
ax.set_ylabel("Amount")
ax.set_xlabel("Rating")
ax.legend(["Rating"])