# %%
import btc_data
from prediction import XGBClassifier, compute_rolling, backtest, precision_score
import pandas as pd
from datetime import datetime


# %%
def main():
    btc = pd.read_csv("btc_data.csv", index_col=0)

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
