import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier


# model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42)

# model = XGBClassifier(random_state=42, learning_rate=0.1, n_estimators=200)

# btc = pd.read_csv("./btc_data.csv", index_col=0)

# train = btc.iloc[:-200]
# test = btc[-200:]
#
# predictors = [
#     "Close",
#    "Volume",
#    "Open",
#    "High",
#    "Low",
#    "edit_count",
#    "sentiment",
#    "neg_sentiment",
# ]
#
# model.fit(train[predictors], train["Target"])
#
# preds = model.predict(test[predictors])
# preds = pd.Series(preds, index=test.index)
#
# precison = precision_score(test["Target"], preds)
#
# print(precison)


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])

    preds = pd.Series(preds, index=test.index, name="predictions")

    combined = pd.concat([test["Target"], preds], axis=1)

    return combined


def backtest(data, model, predictors, start=1095, step=150):
    all_preds = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i : i + step].copy()

        predictions = predict(train, test, predictors, model)

        print(precision_score(predictions["Target"], predictions["predictions"]))

        all_preds.append(predictions)

    return pd.concat(all_preds)


# predictions = backtest(btc, model, predictors)
#
# print(precision_score(predictions["Target"], predictions["predictions"]))


def compute_rolling(btc):
    horizons = [2, 7, 15, 30]

    new_preds = ["Close", "sentiment", "neg_sentiment"]
    for horizon in horizons:
        rolling_averages = btc.rolling(horizon, min_periods=1).mean()

        ratio_col = f"close_ratio_{horizon}"
        btc[ratio_col] = btc["Close"] / rolling_averages["Close"]

        edit_col = f"edit_{horizon}"
        btc[edit_col] = rolling_averages["edit_count"]

        rolling = btc.rolling(horizon, closed="left", min_periods=1).mean()
        trend_column = f"trend_{horizon}"
        btc[trend_column] = rolling["Target"]

        new_preds += [ratio_col, trend_column, edit_col]

    return btc, new_preds


# btc, predictors = compute_rolling(btc.copy())
#
# preds = backtest(btc, model, predictors)
# print(precision_score(preds["Target"], preds["predictions"]))
#
# preds.to_csv("preds.csv")
