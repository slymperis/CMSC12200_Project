import yfinance_ez as yf
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

START_DATE = dt.datetime(1970, 1, 1)
END_DATE = dt.datetime.now()
tckr_set = {"AAPL", "MSFT", "IBM", "DELL", "INTC"}

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

def train_validate_test_split(df):
    """
    Splits DataFrame into 70% train, 15% validation, 15% test sets
    :param df: Pandas DataFrame to split
    :return: train, validation, test DataFrame tuple
    """
    rows = len(df)
    train_rows = int(rows * .7)
    validation_rows = int(rows * .85)
    return df.iloc[:train_rows,:].copy(), df.iloc[train_rows:validation_rows,:].copy(), \
           df.iloc[validation_rows:,:].copy()

def add_feature_to_model(df, features, formula):
    """
    Adds the single best feature to a formula based on BIC
    :param df: Pandas DataFrame
    :param features: List of possible features (Strings) to add
    :param formula: Formula of model to start out with
    :return: model object, formula (String), feature (String) tuple
    """
    min_bic = None
    best_model = None
    best_formula = None
    best_feature = None
    for feature in features:
        new_formula = formula + feature
        fitted_model = sm.ols(formula=new_formula,
                              data=df).fit()
        model_bic = fitted_model.bic
        if (min_bic is None) or (model_bic < min_bic):
            min_bic = model_bic
            best_model = fitted_model
            best_formula = new_formula
            best_feature = feature
    return best_model, best_formula, best_feature

def find_lin_reg_models(df, key_to_predict, max_lags, models_to_return):
    """
    :param df: Pandas DataFrame of log returns
    :param key_to_predict: Pandas column name to predict
    :param max_lags: Maximum number of lags to include for any given column
    :param models_to_return: Number of models to return
    :return: List of BIC, formula, StatsModels linear regression model object tuples
    """
    create_df_lags(df, key_to_predict, max_lags)
    df = df.dropna(how="all", axis=1)
    x_features = list(df.columns)
    x_features.remove(key_to_predict)
    num_features = len(x_features)
    formula = key_to_predict + " ~ "
    model_list = []
    for k in range(1, num_features + 1):
        best_model, formula, best_feature = add_feature_to_model(df, x_features, formula)
        x_features.remove(best_feature)
        model_list.append((best_model.bic, formula, best_model))
        formula += " + "
    model_list = sorted(model_list)
    return model_list[:models_to_return]

def evaluate_model(df, model_tuple, key_to_predict, max_lags):
    """
    Evaluates how well a given model predicts the data
    :param df: Pandas DataFrame
    :param model_tuple: model tuple containing BIC, formula, model object
    :return:
    """
    create_df_lags(df, key_to_predict, max_lags)
    bic, formula, model = model_tuple
    df["preds"] = model.predict(df)
    mse = np.nanmean((df["preds"] - df[key_to_predict])**2)
    return mse

# TODO: Call evaluate model on each element in list of top models to select best model

x = get_yfinance_data(tckr_set)
df = get_log_return_df(x)
train, validate, test = train_validate_test_split(df)
top_models = find_lin_reg_models(train, "AAPL_log_return", 2, 4)

top_model = top_models[0]
print(evaluate_model(validate, top_model, "AAPL_log_return", 2))