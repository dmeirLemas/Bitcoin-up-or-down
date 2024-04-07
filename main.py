# %%
import btc_data
from prediction import XGBClassifier, compute_rolling, backtest, precision_score
import pandas as pd
from datetime import datetime
from transformers import pipeline
import time
from multiprocessing import Pool, cpu_count
from statistics import mean
import pandas as pd
from datetime import datetime
import pickle
import mwclient


def download_new_wiki_data(date: str):
    print("Downloading")

    site = mwclient.Site("en.wikipedia.org")
    page = site.pages["Bitcoin"]

    revs = list(
        site.api(
            "query",
            prop="revisions",
            titles="Bitcoin",
            rvprop="timestamp|comment",
            rvlimit="max",
            rvstart=f"{date}T00:00:00Z",
            rvdir="older",  # Sort revisions from newest to oldest
        )["query"]["pages"].values()
    )[0]["revisions"]

    revs = [rev for rev in revs if rev["timestamp"] != date]

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
    date = rev["timestamp"][:10]

    try:
        comment = rev["comment"]
        sentiment = find_sentiment(comment)
        return date, sentiment
    except:
        return date, 0


def prod(last: str) -> None:
    edits = {}
    wiki = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
    revs = download_new_wiki_data(last)
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

    dates = pd.date_range(start=last, end=datetime.today())

    edits_df = edits_df.reindex(dates, fill_value=0)

    edits_df = pd.concat([wiki.iloc[:-1], edits_df])

    rolling_edits = edits_df.rolling(30).mean()

    rolling_edits = rolling_edits.dropna()

    rolling_edits.to_csv("wikipedia_edits.csv", index="date")


# %%
def main():
    btc = pd.read_csv("btc_data.csv", index_col=0)
    wiki = pd.read_csv("wikipedia_edits.csv", index_col=0)

    if wiki.index[-1] < datetime.today().strftime("%Y-%m-%d"):
        prod(wiki.index[-1])
        wiki = pd.read_csv("wikipedia_edits.csv")

    if btc.index[-1] < datetime.today().strftime("%Y-%m-%d"):
        btc_data.download_new_btc_data()
        btc = pd.read_csv("btc_data.csv", index_col=0)

    model = XGBClassifier(random_state=42, learning_rate=0.1, n_estimators=200)
    btc, predictors = compute_rolling(btc.copy())

    preds = backtest(btc, model, predictors)

    print(precision_score(preds["Target"], preds["predictions"]))
    preds.to_csv("preds.csv")


# %%
if __name__ == "__main__":
    main()
