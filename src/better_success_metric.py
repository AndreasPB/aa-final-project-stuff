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
def month_diff(date) -> int:
    diff = relativedelta.relativedelta(lastest_date, date)
    return diff.months + diff.years * 12


df["monthDiff"] = df["reviewTime"].apply(month_diff)
df["monthDiff"].head()

# %%
def calc_vote_success(month_diff: int, votes: int) -> float:
    if month_diff:
        return votes / month_diff
    return votes


df["voteSuccess"] = df[["monthDiff", "vote"]].apply(
    lambda x: calc_vote_success(*x), axis=1
)

df.sort_values("voteSuccess").tail()

# %%
