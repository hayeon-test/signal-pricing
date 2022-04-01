import logging
import os
from time import sleep

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.NOTSET
)

SYMBOL = os.getenv("SYMBOL")
INTERVAL = os.getenv("INTERVAL")
DATAPATH = "/data"
LAG_SLICES = ["year1month1"]
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_intraday_url(symbol, interval, lag_slice, apikey):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice={}&adjusted=false&apikey={}".format(
        symbol, interval, lag_slice, apikey
    )
    return url


def url_to_df(url):
    df = pd.read_csv(url)
    return df


def df_to_parquet(df, symbol, interval, lag_slice):
    start = df.time.min().replace(" ", "-").replace(":", "-")
    end = df.time.max().replace(" ", "-").replace(":", "-")
    fname = os.path.join(
        DATAPATH,
        "raw",
        "{}_{}_{}_{}.parquet.gzip".format(symbol, interval, lag_slice, start, end),
    )
    df.to_parquet(path=fname, engine="pyarrow", compression="gzip", index=False)
    logging.info("Saved as {}".format(fname))


def fetch_to_parquet(
    lag_slice, symbol=SYMBOL, interval=INTERVAL, apikey=ALPHA_VANTAGE_API_KEY
):
    url = fetch_intraday_url(symbol, interval, lag_slice, apikey)
    df = url_to_df(url)
    df_to_parquet(df, symbol, interval, lag_slice)


if __name__ == "__main__":
    for LAG_SLICE in LAG_SLICES:
        fetch_to_parquet(LAG_SLICE)
        # sleep(21) # API key is rate limited at 5/min, 500/day if using a free key.
