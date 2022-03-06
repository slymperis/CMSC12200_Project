import yfinance_ez as yf
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

START_DATE = dt.datetime(1970, 1, 1)
END_DATE = dt.datetime.now()

POSITIVE = {"long-term buy", "positive", "outperform", "market outperform", "buy", "overweight", "strong buy",
            "sector outperform"}
NEUTRAL = {"neutral", "fair value", "sector perform", "equal-weight", "hold", "equal-weight", "peer perform",
           "sector weight", "market perform", "perform"}
NEGATIVE = {"negative", "underweight", "sell", "reduce", "underperform"}

def create_df_cols(df):
    """
    Takes a Pandas DataFrame and creates log return column
    Inputs: df (DataFrame)
    :return: Modified DataFrame
    """
    df["log_return"] = np.log(df["Close"]).diff()
    df = df.dropna()
    return df

def get_yfinance_data(ticker_set):
    """
    Takes a set of tickers and returns a dictionary mapping tickers to OHLCV DataFrames
    Inputs: Set of tickers (Strings)
    :return: Dictionary
    """
    output_dict = {}
    for ticker in ticker_set:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.get_history(period=yf.TimePeriods.Day,
                                      start=START_DATE, end=END_DATE)
        hist = create_df_cols(hist)
        output_dict[ticker] = hist
    return output_dict

def get_log_return_df(d):
    """
    If only interested in log returns, turns dictionary mapping tickers to data into single
    DataFrame just including log returns
    :return:
    """
    col_names = []
    series = []
    for ticker, df in d.items():
        col_names.append(ticker+"_log_return")
        log_return_series = df["log_return"]
        series.append(log_return_series)
    combined_df = pd.concat(series, keys=col_names, axis=1)
    return combined_df

def create_df_lags(df, key_to_predict, max_lags):
    """
    Creates DataFrame columns up to maximum number of lags
    :param df: DataFrame to modify
    :param key_to_predict: Key not to modfiy
    :return: None, modifies DataFrame in place
    """
    for col in df.columns:
        lag = 1
        for i in range(max_lags):
            df[col+"_lag_"+str(lag)] = df[col].shift(lag)
            lag += 1
        if col != key_to_predict:
            df.drop(col, axis=1, inplace=True)

def get_analyst_recommendations(tckr):
    """
    Pulls Yahoo finance data on analyst recommendations by date (i.e "buy", "sell", "hold"),
    scores them as positive (1) or negative (-1) and then for each day since the first available
    analyst recommendation calculates the rolling average of the last 90 days of analyst scores
    :param tckr: (String)
    :return: DataFrame
    """
    ticker = yf.Ticker(tckr)
    recs = ticker.recommendations
    scores = []
    for rating in recs["To Grade"]:
        rating = rating.lower()
        if rating in POSITIVE:
            score = 1
        elif rating in NEGATIVE:
            score = -1
        else:
            score = 0
        scores.append(score)
    recs["scores"] = scores
    recs["date"] = pd.to_datetime(recs.index).date
    recs = recs.set_index("date")
    start_date = recs.index[0]
    end_date = recs.index[-1]
    print(start_date, end_date)
    total_days = (end_date - start_date).days
    date_list = []
    for day in range(total_days+1):
        date_list.append((start_date + dt.timedelta(days=day)))
    date_df = pd.DataFrame(date_list, columns=["Date"])
    date_df = date_df.set_index("Date")
    recs = pd.DataFrame(recs["scores"], columns=["scores"])
    recs_sum = recs.groupby(recs.index)["scores"].sum()
    recs_counts = recs.groupby(recs.index)["scores"].count()
    merged_df = pd.merge(date_df, recs_sum, left_index=True, right_index=True, how="outer")
    merged_df = pd.merge(merged_df, recs_counts, left_index=True, right_index=True, how="outer")
    merged_df = merged_df.rename(columns={"scores_x": "sum", "scores_y": "count"})
    sum_roll = merged_df["sum"].rolling(90, 0).apply(lambda x: np.nansum(x))
    count_roll = merged_df["count"].rolling(90, 0).apply(lambda x: np.nansum(x))
    merged_df["rolling_avg"] = sum_roll/count_roll
    merged_df = merged_df[90:]
    merged_df["analyst_recs_lagged"] = merged_df["rolling_avg"].shift(1)
    return merged_df
