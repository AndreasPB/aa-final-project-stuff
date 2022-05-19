# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("magazine_reviews_cleanup.tsv", sep='\t')
df

# %%
bins = list(range(0, 50+25, 10))
df["vote"].value_counts(bins=bins).plot(figsize=(9, 3),kind="bar", rot=0)
plt.ylabel("Amount of reviews")
plt.xlabel("Vote")
plt.legend(["Review votes"])

# %%
bins = list(range(1, 50+25, 10))
df["vote"].value_counts(bins=bins).plot(figsize=(9, 3),kind="bar", rot=0)
plt.ylabel("Amount of reviews")
plt.xlabel("Vote")
plt.legend(["Review votes"])
# %%
votes = df["vote"].value_counts()
no_votes = votes[0].sum()
has_votes = votes[1:].sum()
plt.bar(["No votes", "Votes"], [no_votes, has_votes])
plt.ylabel("Amount of reviews")
plt.xlabel("Vote")

plt.legend(["Review votes"])
# %%
# for loops?
bins = list(range(1, 50+25, 10))
df["vote"].value_counts(bins=bins)
print("I AM HERE ISAJDIOASJDIOAJIODJIAOJDIOAJOID")
# %%
#data = df["vote"].value_counts(bins=bins)
five_star_reviews = df.loc[df.rating == 5]
print(type(five_star_reviews))
four_star_reviews = df.loc[df.rating == 4]
three_star_reviews = df.loc[df.rating == 3]
two_star_reviews = df.loc[df.rating == 2]
one_star_reviews = df.loc[df.rating == 1]
five = five_star_reviews["vote"].value_counts(bins=bins)
four = four_star_reviews["vote"].value_counts(bins=bins)
three = three_star_reviews["vote"].value_counts(bins=bins)
two = two_star_reviews["vote"].value_counts(bins=bins)
one = one_star_reviews["vote"].value_counts(bins=bins)

print("THIS IS NOT FUNNNNY: ", one.to_list())

mogens = np.array([five.to_list(),
                   four.to_list(),
                   three.to_list(),
                   two.to_list(),
                   one.to_list()])

print(mogens)

graeder = ["[1:11]", "[11:21]", "[21:31]", "[31:41]","[41:51]", "[51:61]", "[61:71]"]
tuder = range(5, 0, -1)

fig,ax = plt.subplots()
im = ax.imshow(mogens)
ax.set_yticks(range(len(tuder)), lables="rating")
ax.set_xticks(range(len(graeder)), lables="bins")
ax.set_ylabel("Rating")
ax.set_xlabel("Bins")
ax.set_xticklabels(graeder)
ax.set_yticklabels(tuder)

print("BINS: ", len(bins))

for i in range(5):
    for j in range(len(bins)-1):
        text = ax.text(j, i, mogens[i, j], ha="center", va="center")

fig.tight_layout()
plt.show()
# %%
