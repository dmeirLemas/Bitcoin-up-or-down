from transformers import pipeline
import time
from multiprocessing import Pool, cpu_count
from statistics import mean
import pandas as pd
from datetime import datetime
import pickle
import mwclient


def download_new_wiki_data() -> None:
    print("Downloading")

    site = mwclient.Site("en.wikipedia.org")
    page = site.pages["Bitcoin"]

    revs = list(page.revisions())

    revs = sorted(
        revs,
        key=lambda rev: rev["timestamp"],
    )

    with open("bitcoin_revisions.pkl", "wb") as f:
        pickle.dump(revs, f)

    return revs


def load_wiki_data():
    with open("bitcoin_revisions.pkl", "rb") as f:
        revs = pickle.load(f)

    return revs


# Delete the model="sentiment-analysis-model"  to download the pretrained model.
# Then you can save it using pickle
sentiment_pipeline = pipeline("sentiment-analysis", model="sentiment-analysis-model")


def find_sentiment(text):
    sent = sentiment_pipeline([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1
    return score


def process_revision(rev):
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    print(date)

    try:
        comment = rev["comment"]
        sentiment = find_sentiment(comment)
        return date, sentiment
    except:
        return date, 0


def prod():
    edits = {}

    revs = download_new_wiki_data()
    # revs = load_wiki_data()

    with Pool(cpu_count()) as pool:
        results = pool.map(process_revision, revs)

    for date, sentiment in results:
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)

        edits[date]["edit_count"] += 1
        edits[date]["sentiments"].append(sentiment)

    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len(
                [i for i in edits[key]["sentiments"] if i < 0]
            ) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0

        del edits[key]["sentiments"]

    edits_df = pd.DataFrame.from_dict(edits, orient="index")

    edits_df.index = pd.to_datetime(edits_df.index)

    dates = pd.date_range(start="2009-03-08", end=datetime.today())

    edits_df = edits_df.reindex(dates, fill_value=0)

    rolling_edits = edits_df.rolling(30).mean()

    rolling_edits = rolling_edits.dropna()

    rolling_edits.to_csv("wikipedia_edits.csv")


if __name__ == "__main__":
    prod()
