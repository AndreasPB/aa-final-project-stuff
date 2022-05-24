# %%
from dateutil import parser
from dateutil import relativedelta

import pandas as pd

# %%
df = pd.read_csv("../data/cleaned_reviews.tsv", sep="\t")
df.head()

# %%
df["reviewTime"] = df["reviewTime"].apply(parser.parse)

# %%
# Sorts accendingly
lastest_date = list(df["reviewTime"].sort_values())[-1]
lastest_date

# %%
def quater_diff(date) -> int:
    diff = relativedelta.relativedelta(lastest_date, date)
    quarters = diff.years * 4
    for month in range(1, diff.months + 1):
        if month % 3 == 0:
            quarters += 1

    return quarters


df["quaterDiff"] = df["reviewTime"].apply(quater_diff)
df["quaterDiff"].head()

# %%
def calc_vote_success(month_diff: int, votes: int) -> float:
    soften_success = 2  # Helps soften the division jump
    return votes / (month_diff + soften_success)


df["voteSuccess"] = df[["quaterDiff", "vote"]].apply(
    lambda x: calc_vote_success(*x), axis=1
)

df.sort_values("voteSuccess").tail()

# %%
