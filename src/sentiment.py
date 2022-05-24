# %%
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
sentiments = SentimentIntensityAnalyzer()

data = pd.read_csv("../data/cleaned_reviews.tsv", sep="\t")
print(data.head())

# %%
data = data.dropna()

ratings = data["rating"].value_counts()
numbers = ratings.index
quantity = ratings.values

custom_colors = ["skyblue", "yellowgreen", 'tomato', "blue", "red"]
plt.figure(figsize=(10, 8))
plt.pie(quantity, labels=numbers, colors=custom_colors)
central_circle = plt.Circle((0, 0), 0.5, color='white')
fig = plt.gcf()
fig.gca().add_artist(central_circle)
plt.rc('font', size=12)
plt.title("Ratings", fontsize=20)
plt.show()

# %%
sentiments = SentimentIntensityAnalyzer()

data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["reviewText"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["reviewText"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["reviewText"]]
print(data.head())

# %%
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])

def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        print("Positive")
        print(f"Score: {a}")
    elif (b>a) and (b>c):
        print("Negative")
        print(f"Score: {b}")
    else:
        print("Neutral")
        print(f"Score: {c}")
sentiment_score(x, y, z)


