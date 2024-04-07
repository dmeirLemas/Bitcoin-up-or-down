import yfinance as yf
import pandas as pd


def download_new_btc_data() -> None:
    btc_ticker = yf.Ticker("BTC-USD")
    btc = btc_ticker.history(period="max")

    btc.index = pd.to_datetime(btc.index)
    btc.index = btc.index.tz_localize(None)

    del btc["Stock Splits"]
    del btc["Dividends"]

    wiki = pd.read_csv("./wikipedia_edits.csv", index_col=0, parse_dates=True)

    btc = btc.merge(wiki, left_index=True, right_index=True)

    btc["Tommorow"] = btc["Close"].shift(-1)
    btc["Target"] = (btc["Tommorow"] > btc["Close"]).astype(int)

    btc.index._name = "Date"

    btc.to_csv("./btc_data.csv")


if __name__ == "__main__":
    download_new_btc_data()
