# %%
import pandas as pd
import matplotlib.pyplot as plt


# %%
EXPORT_PLOTS = False
df = pd.read_csv("../../data/cleaned_reviews.tsv", sep="\t")

# %%
ax = df["rating"].value_counts().sort_index().plot(kind="bar", rot=0)
ax.set_ylabel("Amount")
ax.set_xlabel("Rating")
ax.legend(["Rating"])
if EXPORT_PLOTS:
    plt.savefig("../../report/img/rating_distribution.svg", bbox_inches="tight")
