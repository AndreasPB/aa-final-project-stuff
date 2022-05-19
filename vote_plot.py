# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("magazine_reviews_cleanup.tsv", sep='\t')
df

# %%
bins = pd.cut(df["vote"], list(range(0, 100, 10)))
df.groupby(bins)["vote"].head(500).value_counts().plot(kind="bar", rot=0)
plt.ylabel("Amount")
plt.xlabel("Vote")
plt.legend(["Review votes"])


# %%