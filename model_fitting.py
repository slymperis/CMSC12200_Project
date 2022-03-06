import yfinance_ez as yf
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from yfinance_scraper import create_df_cols, get_yfinance_data, get_log_return_df, create_df_lags, get_analyst_recommendations
from trends import daily_interest_table
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_columns', None)

START_DATE = dt.datetime(1970, 1, 1)
END_DATE = dt.datetime.now()

def get_data(tckr_set, lags, key_to_predict, keywords=None, start_year = 2004, start_month = 1,
             end_year = None, end_month = None, analyst_recs = False):
    """
    Gets lag return DataFrame for a set of tickers and merges with Google trends data on keywords
    Makes sure to properly handle lags of Google trends data on dates that might not be included in the
    Yahoo Finance DataFrame (missing weekends and other non-trading days)
    :param tckr_set: Set of strings (tickers) to get return data for
    :param lags: Number of lags to create
    :param key_to_predict: String, the only key where we keep the contemporaneous value
    :param keywords: Optional, list of strings, Google trends keywords, if None will only get return data
    :param end_year: (Int) Optional, most recent year to get trends data for, if none will pull data up to today
    :param end_month: (Int) Optional, most recent month to get trends data for, if none will pull data up to today
    :param analyst_recs: (Boolean) whether or not to include scored analyst recommendations
    :return: DataFrame
    """
    ret_df = get_log_return_df(get_yfinance_data(tckr_set))
    create_df_lags(ret_df, key_to_predict, lags)
    if keywords is not None:
        if (end_year is None) or (end_month is None):
            today = dt.datetime.now()
            interest_df = daily_interest_table(keywords, today.year, today.month, start_year=start_year,
                                               start_month=start_month)
        else:
            interest_df = daily_interest_table(keywords, end_year, end_month,
                                               start_year=start_year, start_month=start_month)
        for lag in range(1, lags+1):
            date_col_name = "date_lag_" + str(lag)
            ret_df[date_col_name] = ret_df.index
            ret_df[date_col_name] = ret_df[date_col_name] - dt.timedelta(days=lag)
            lag_interest = interest_df[interest_df.index.isin(ret_df[date_col_name])]
            lag_interest_colnames = list(interest_df.columns)
            lag_interest_colnames = [x.replace(" ", "")+"_interest_lag_"+str(lag) for x in lag_interest_colnames]
            lag_interest.columns = lag_interest_colnames
            ret_df = pd.merge(ret_df, lag_interest, left_on=date_col_name, right_index=True)
            ret_df = ret_df.drop(date_col_name, axis=1)
    if analyst_recs:
        for tckr in tckr_set:
            recs = get_analyst_recommendations(tckr)["analyst_recs_lagged"]
            ret_df = pd.merge(ret_df, recs, left_index=True, right_index=True)
            ret_df = ret_df.rename(columns={"analyst_recs_lagged": str(tckr)+"_analyst_recs"})
    return ret_df

#get_data({"AAPL", "MSFT"}, 2, "AAPL_log_return", analyst_recs=True, keywords=["Apple stock"],
 #              start_year=2012, start_month=1, end_year=2014, end_month=1).to_csv("random_forest_test_data.csv")

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

def find_lin_reg_models(df, key_to_predict, max_lags, models_to_return, max_features=None):
    """
    :param df: Pandas DataFrame of log returns
    :param key_to_predict: Pandas column name to predict
    :param max_lags: Maximum number of lags to include for any given column
    :param models_to_return: Number of models to return
    :return: List of BIC, formula, StatsModels linear regression model object tuples
    """
    #create_df_lags(df, key_to_predict, max_lags)
    df = df.dropna(how="all", axis=1)
    x_features = list(df.columns)
    print(key_to_predict)
    if key_to_predict in x_features:
        x_features.remove(key_to_predict)
    if max_features is None:
        feature_limit = len(x_features) + 1
    else:
        feature_limit = max_features
    formula = key_to_predict + " ~ "
    model_list = []
    for k in range(1, feature_limit):
        best_model, formula, best_feature = add_feature_to_model(df, x_features, formula)
        x_features.remove(best_feature)
        model_list.append((best_model.bic, formula, best_model))
        formula += " + "
    model_list = sorted(model_list)
    return model_list[:models_to_return]

def evaluate_model(df, model_tuple, key_to_predict):
    """
    Evaluates how well a given model predicts the data
    :param df: Pandas DataFrame
    :param model_tuple: model tuple containing BIC, formula, model object
    :param key_to_predict: the column we want to predict
    :return:
    """
    bic, formula, model = model_tuple
    df["preds"] = model.predict(df)
    mse = np.nanmean((df["preds"] - df[key_to_predict])**2)
    return mse

def evaluate_models(df, model_tuples, key_to_predict, max_lags):
    """
    Evaluates how well a list of models predicts the data and returns the model with smallest MSE
    :param df: Pandas DataFrame
    :param model_tuples: model tuple containing BIC, formula, model object
    :param key_to_predict: the column we want to predict
    :param max_lags: the maximum number of lags to create
    :return: the best model tuple
    """
    #create_df_lags(df, key_to_predict, max_lags)
    best_mse = None
    best_model_tup = None
    for model_tup in model_tuples:
        mse = evaluate_model(df, model_tup, key_to_predict)
        if best_mse is None:
            best_mse = mse
            best_model_tup = model_tup
        else:
            if mse < best_mse:
                best_mse = mse
                best_model_tup = model_tup
    return best_model_tup

def get_best_random_forest(train, validate, key_to_predict, min_depth=1, max_depth=5):
    train, validate = train.fillna(0), validate.fillna(0)
    X_train = train.loc[:, train.columns != key_to_predict]
    Y_train = train[key_to_predict]
    X_validate = validate.loc[:, validate.columns != key_to_predict]
    Y_validate = validate[key_to_predict]
    num_features = int(len(train.columns) ** .5)
    best_model, best_mse = None, None
    for depth in range(min_depth, max_depth):
        rf = RandomForestRegressor(n_estimators=300, max_depth=depth, max_features=num_features)
        rf.fit(X_train, Y_train)
        preds = rf.predict(X_validate)
        validate_copy = validate.copy(deep=True)
        validate_copy["preds"] = preds
        validate_copy["error"] = validate_copy[key_to_predict] - validate_copy["preds"]
        mse = np.average((validate_copy["error"] ** 2))
        if (best_mse is None) or (mse < best_mse):
            best_model = rf
    return rf

def back_test_model(df, model_tuple, key_to_predict, max_lags):
    """
    With the selected "best" model backtests on the test set to check its returns
    :param df: Pandas DataFrame
    :param model_tuple: model tuple containing BIC, formula, model object
    :param key_to_predict: the column to predict
    :param max_lags: the maximum number of lags the model needs
    :return:
    """
    df["position"] = np.where(df["preds"] > 0, 1, -1)
    df["strat_return"] = df["position"] * df[key_to_predict]
    total_log_return = np.sum(df["strat_return"])
    average_return = np.average(df["strat_return"])
    std = np.std(df["strat_return"])
    sharpe = (average_return/std) * (365 ** .5)
    total_underlying_log_return = np.sum(df[key_to_predict])
    average_underlying_return = np.average(df[key_to_predict])
    underlying_std = np.std(df[key_to_predict])
    underlying_sharpe = (average_underlying_return/underlying_std) * (365 ** .5)
    """
    print("Total log return: ", total_log_return)
    print("Average return: ", average_return)
    print("Sharpe: ", sharpe)
    print("Total underlying log return: ", total_underlying_log_return)
    print("Average underlying return: ", average_underlying_return)
    print("Underlying sharpe: ", underlying_sharpe)
    """
    output = {"total_log_return": total_log_return,
              "average_return": average_return,
              "sharpe": sharpe, "total_underlying_log_return": total_underlying_log_return,
              "average_underlying_return": average_underlying_return,
              "underlying_sharpe": underlying_sharpe}
    return output

def giga_function(tckr_set, key_to_predict, max_lags, models_to_evaluate):
    data = get_yfinance_data(tckr_set)
    data = get_log_return_df(data)
    train, validate, test = train_validate_test_split(data)
    top_models = find_lin_reg_models(train.copy(deep=True), key_to_predict, max_lags, models_to_evaluate)
    validated_model = evaluate_models(validate.copy(deep=True), top_models, key_to_predict, max_lags)
    bic, formula, model = validated_model
    y = test.copy(deep=True)
    y["preds"] = model.predict(y)
    backtest_results = back_test_model(y, validated_model, key_to_predict, max_lags)
    return validated_model, backtest_results

def main(tckr_set, lags, key_to_predict, models_to_evaluate, keywords=None, start_year = 2004, start_month = 1,
             end_year = None, end_month = None, analyst_recs = False):
    data = get_data(tckr_set, lags, key_to_predict, keywords, start_year, start_month, end_year, end_month,
                    analyst_recs)
    train, validate, test = train_validate_test_split(data)
    top_models = find_lin_reg_models(train.copy(deep=True), key_to_predict, lags, models_to_evaluate)
    validated_model = evaluate_models(validate.copy(deep=True), top_models, key_to_predict, lags)
    bic, formula, model = validated_model
    test_copy = test.copy(deep=True)
    test_copy["preds"] = model.predict(test_copy)
    backtest_results = back_test_model(test_copy, validated_model, key_to_predict, lags)
    print(backtest_results)

    rf = get_best_random_forest(train, validate, key_to_predict)
    test_copy = test.copy(deep=True).fillna(0)
    test_copy["preds"] = rf.predict(test_copy.loc[:, test_copy.columns != key_to_predict])
    backtest_results = back_test_model(test_copy, validated_model, key_to_predict, lags)
    print(backtest_results)

main({"AAPL", "MSFT"}, 2, "AAPL_log_return", 2, ["Apple stock"], 2013, 1, 2021, 5, True)

#data = pd.read_csv("testing_data.csv")
#data = data.set_index("Date")
#train, validate, test = train_validate_test_split(data)
"""
GET BEST LINEAR MODEL
print(train)
print(validate)
top_models = find_lin_reg_models(train, "AAPL_log_return", 3, 3)
validated_model = evaluate_models(validate, top_models, "AAPL_log_return", 3)
backtest_results = back_test_model(test, validated_model, "AAPL_log_return", 3)
print(backtest_results)
print(validated_model)
"""

# random forest: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0


#get_data({"AAPL", "MSFT"}, 3, "AAPL_log_return", keywords=["Apple stock", "Microsoft stock"], end_year=2005, end_month=1).to_csv("testing_data.csv")

"""
tckr_set = {"AAPL", "MSFT"}
keywords = ["Apple stock"]
ret_df = get_log_return_df(get_yfinance_data(tckr_set))
ret_df["date_copy"] = ret_df.index
ret_df["date_copy"] = pd.to_datetime(ret_df["date_copy"])
ret_df["date_lag_1"] = ret_df["date_copy"] - dt.timedelta(days=1)

interest_df = daily_interest_table(keywords, 2004, 7)
print(ret_df)
print(interest_df)
"""
# could just merge on date but would lose google trends info on weekends...
# could also create lags of date column and then merge in masked google trends on that column