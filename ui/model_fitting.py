import yfinance_ez as yf
import datetime as dt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from yfinance_scraper import create_df_cols, get_yfinance_data, get_log_return_df, create_df_lags, \
    get_analyst_recommendations
from trends import daily_interest_table
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

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
    :return: ret_df, oos_df tuple where ret_df is for fitting and testing models and oos_df is exclusively for
    forecasting tomorrow's expected return
    """
    ret_df = get_log_return_df(get_yfinance_data(tckr_set))
    oos_df = ret_df.copy(deep=True)[-100:]
    create_df_lags(ret_df, key_to_predict, lags)
    create_df_lags(oos_df, key_to_predict, lags, oos=True)
    if keywords is not None: # gets keywords from Google trends
        if (end_year is None) or (end_month is None):
            today = dt.datetime.now()
            interest_df = daily_interest_table(keywords, today.year, today.month, start_year=start_year,
                                               start_month=start_month)
        else:
            interest_df = daily_interest_table(keywords, end_year, end_month,
                                               start_year=start_year, start_month=start_month)
        for lag in range(1, lags+1): # merges in previous days Google trends depending on how many lags we want
            date_col_name = "date_lag_" + str(lag) # this is necessary because financial data skips certain days (weekends) but we want the Google trends lag to be able to be on a weekend
            ret_df[date_col_name] = ret_df.index
            ret_df[date_col_name] = ret_df[date_col_name] - dt.timedelta(days=lag)
            lag_interest = interest_df[interest_df.index.isin(ret_df[date_col_name])]
            lag_interest_colnames = list(interest_df.columns)
            lag_interest_colnames = [x.replace(" ", "")+"_interest_lag_"+str(lag) for x in lag_interest_colnames]
            lag_interest.columns = lag_interest_colnames
            ret_df = pd.merge(ret_df, lag_interest, left_on=date_col_name, right_index=True, how="outer")
            ret_df = ret_df.dropna(axis=1, how="all")
            ret_df = ret_df.drop(date_col_name, axis=1)
        for lag in range(0, lags): # merges in previous days Google trends into the delagged DataFrame that will be used for forecasting tomorrow's portfolio
            date_col_name = "date_lag_" + str(lag)
            oos_df[date_col_name] = oos_df.index
            oos_df[date_col_name] = oos_df[date_col_name] - dt.timedelta(days=lag)
            needed_interest = interest_df[interest_df.index.isin(oos_df[date_col_name])]
            needed_interest_colnames = list(interest_df.columns)
            needed_interest_colnames = [x.replace(" ", "")+"_interest_lag_"+str(lag+1) for x in needed_interest_colnames]
            needed_interest.columns = needed_interest_colnames
            oos_df = pd.merge(oos_df, needed_interest, left_on=date_col_name, right_index=True, how="outer")
            oos_df = oos_df.dropna(axis=1, how="all")
            oos_df = oos_df.drop(date_col_name, axis=1)
    if analyst_recs: # merges in analyst recommendations
        for tckr in tckr_set:
            recs = get_analyst_recommendations(tckr)["rolling_avg"]
            ret_df = pd.merge(ret_df, recs, left_index=True, right_index=True, how="outer")
            oos_df = pd.merge(oos_df, recs, left_index=True, right_index=True, how="outer")
            ret_df = ret_df.dropna(axis=1, how="all")
            oos_df = oos_df.dropna(axis=1, how="all")
            ret_df["rolling_avg"] = ret_df["rolling_avg"].shift(1)
            ret_df = ret_df.rename(columns={"rolling_avg": str(tckr)+"_analyst_recs"})
            oos_df = oos_df.rename(columns={"rolling_avg": str(tckr)+"_analyst_recs"})
    return ret_df, oos_df

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
    """
    Gets the best random forest by trying model's of various depth and picking the one with the best performance
    in the validation set
    :param train: Train data
    :param validate: Validate data
    :param key_to_predict: String
    :param min_depth: Int
    :param max_depth: Int
    :return: Best Random Forest model
    """
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

def get_best_gradient_booster(train, validate, key_to_predict, min_depth = 1, max_depth=5):
    """
    Gets the best Gradient Booster model by trying model's of various depth and evaluating performance on the
    validation set
    :param train: Train data
    :param validate: Validate data
    :param key_to_predict: String
    :param min_depth: Int
    :param max_depth: Int
    :return: Best Gradient Booster model
    """
    train, validate = train.fillna(0), validate.fillna(0)
    X_train = train.loc[:, train.columns != key_to_predict]
    Y_train = train[key_to_predict]
    X_validate = validate.loc[:, validate.columns != key_to_predict]
    Y_validate = validate[key_to_predict]
    best_model, best_mse = None, None
    for depth in range(min_depth, max_depth):
        gb = GradientBoostingRegressor(max_depth = depth)
        gb.fit(X_train, Y_train)
        preds = gb.predict(X_validate)
        validate_copy = validate.copy(deep=True)
        validate_copy["preds"] = preds
        validate_copy["error"] = validate_copy[key_to_predict] - validate_copy["preds"]
        mse = np.average((validate_copy["error"] ** 2))
        if (best_mse is None) or (mse < best_mse):
            best_model = gb
    return gb

def back_test_model(df, key_to_predict, max_lags):
    """
    With the selected "best" model backtests on the test set to check its returns
    :param df: Pandas DataFrame
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
    output = {"total_log_return": total_log_return,
              "average_return": average_return,
              "sharpe": sharpe, "total_underlying_log_return": total_underlying_log_return,
              "average_underlying_return": average_underlying_return,
              "underlying_sharpe": underlying_sharpe}
    return output

def get_best_oos_fcast(tckr_set, lags, key_to_predict, models_to_evaluate, keywords=None, start_year = 2004, start_month = 1,
             end_year = None, end_month = None, analyst_recs = False):
    model_results = []
    data, oos_data = get_data(tckr_set, lags, key_to_predict, keywords, start_year, start_month, end_year, end_month,
                    analyst_recs)
    """
    For a single ticker gets data and fits best linear, Random Forest, and Gradient Booster model before
    choosing the one that generates the highest Sharpe ratio in the test set and using it to forecast
    expected returns one day out of sample
    :params: Data specifications passed to get_data
    :return: expected return forecast (float)
    """
    train, validate, test = train_validate_test_split(data)
    test_final = test.copy(deep=True)
    top_models = find_lin_reg_models(train.copy(deep=True), key_to_predict, lags, models_to_evaluate)
    validated_model = evaluate_models(validate.copy(deep=True), top_models, key_to_predict, lags)
    bic, formula, model = validated_model
    test_copy = test.copy(deep=True)
    test_copy["preds"] = model.predict(test_copy)
    backtest_results = back_test_model(test_copy, key_to_predict, lags)
    lin_sharpe = backtest_results["sharpe"]
    model_results.append((lin_sharpe, model))

    rf = get_best_random_forest(train, validate, key_to_predict)
    test_copy = test.copy(deep=True).fillna(0)
    test_copy["preds"] = rf.predict(test_copy.loc[:, test_copy.columns != key_to_predict])
    backtest_results = back_test_model(test_copy, key_to_predict, lags)
    rf_sharpe = backtest_results["sharpe"]
    model_results.append((rf_sharpe, rf))

    gb = get_best_gradient_booster(train, validate, key_to_predict)
    test_copy = test.copy(deep=True).fillna(0)
    test_copy["preds"] = gb.predict(test_copy.loc[:, test_copy.columns != key_to_predict])
    backtest_results = back_test_model(test_copy, key_to_predict, lags)
    gb_sharpe = backtest_results["sharpe"]
    model_results.append((gb_sharpe, gb))

    best_model_tup = max(model_results)
    best_sharpe, best_model = best_model_tup
    return best_model.predict(oos_data.drop(key_to_predict, axis=1).fillna(0))[-1]

def get_expected_returns(model_specs):
    """
    Takes in a dictionary mapping tickers to requested model specifications and tries different linear,
    Random Forest, and Gradient Booster models to model each ticker's returns
    Outputs an array with the best forecast of the one step out of sample expected return for each ticker as well
    as log returns data to be used for modelling the variance-covariance matrix
    :param model_specs: Dictionary
    :return: expected_returns (NumPy array), log_returns (DataFrame) tuple
    """
    expected_returns = []
    tckrs = set(model_specs.keys())
    log_returns = get_log_return_df(get_yfinance_data(sorted(tckrs)))
    print(log_returns.dropna())
    for tckr, model_spec in sorted(model_specs.items()):
        tckr_set, lags, key_to_predict, models_to_evaluate, keywords, start_year, start_month,\
        end_year, end_month, analyst_recs = model_spec
        expected_ret = get_best_oos_fcast(tckr_set, lags, key_to_predict, models_to_evaluate, keywords=keywords, start_year=start_year,
                           start_month=1, end_year=end_year, end_month=end_month, analyst_recs=analyst_recs)
        expected_returns.append(expected_ret)
    num_assets = len(expected_returns)
    expected_returns = np.array(expected_returns).reshape(num_assets, -1).T
    return expected_returns, log_returns

