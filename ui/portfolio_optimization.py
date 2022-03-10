import pandas as pd
import numpy as np
import random
import itertools
import datetime as dt
from itertools import combinations
from correlation_modelling import forecast_var_cov_matrix
from model_fitting import get_expected_returns
from itertools import permutations

def get_var_covar_matrix(df):
    """
    Calls the function in correlation_modelling.py that uses R to forecast the variance-covariance matrix
    :param df: DataFrame of log returns
    :return: NumPy matrix
    """
    var_cov_matrix = np.matrix(forecast_var_cov_matrix(df))
    return var_cov_matrix

def get_portfolio_sharpe(var_cov_matrix, expected_returns, weights):
    """
    Calculates the sharpe ratio given expected returns, a variance-covariance matrix, and a 1 dimensional
    array of weights
    :param var_cov_matrix: Forecast of the variance-covariance matrix (NumPy matrix)
    :param expected_returns: NumPy array of expected returns
    :param weights: NumPy array of portfolio weights
    :return: Sharpe ratio (float)
    """
    variance = (weights.T * var_cov_matrix * weights)
    sd = float(np.sqrt(variance))
    expected_returns = np.exp(expected_returns) - 1
    portfolio_er = float(expected_returns @ weights)
    sharpe = portfolio_er/sd
    return sharpe, portfolio_er, sd

def get_random_weights(num_assets):
    """
    Generates random portfolio weights that sum to 1
    :param num_assets: Number of weights to generate
    :return: Weights (NumPy array)
    """
    weights = np.random.random(num_assets) * 2 - 1
    weights /= weights.sum()
    weights = np.array(weights).reshape(num_assets, -1)
    if np.any(weights > 1) or np.any(weights < -1):
        return get_random_weights(num_assets)
    else:
        return weights

def portfolio_weights_monte_carlo(df, expected_returns, var_cov_matrix, iterations=1000000):
    """
    Runs a Monte Carlo simulation to approximate optimal portfolio weights for an arbitrary number of assets
    :param df: DataFrame of log returns
    :param expected_returns: Array of expected returns
    :param var_cov_matrix: Forecasted variance-covariance matrix
    :param iterations: Number of random weights to try (default is 1 million, takes about 40 seconds to run)
    :return: optimal weights (array), optimal portfolio expected return (Float), optimal portfolio
     expected standard deviation (Float) tuple
    """
    num_assets = var_cov_matrix.shape[1]
    best_sharpe, best_weights = None, None
    for i in range(iterations):
        weights = get_random_weights(num_assets)
        sharpe, er, sd = get_portfolio_sharpe(var_cov_matrix, expected_returns, weights)
        if best_sharpe is None:
            best_sharpe = sharpe
            best_weights = weights
        elif sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    s, portfolio_er, portfolio_sd = get_portfolio_sharpe(var_cov_matrix, expected_returns, best_weights)
    return best_weights, portfolio_er, portfolio_sd

def main(model_specs, iterations=1000000):
    """
    Takes a dictionary mapping tickers to a tuple containing a set of ticker data to regress on,
    a number of lags to use, a key to predict, a number of models to evaluate, a list of keywords to query
    Google trends for, a start year, start month, end year, end month, and a boolean on whether or not to include
    analyst recommendations
    Fits models to best forecast expected returns and then forecasts the one day out of sample expected returns
    matrix as well as the conditional variance-covariance matrix, approximates optimal portfolio weights and returns
    them as a numpy array
    :param model_specs: Dictionary
    :param iterations: Optional, number of Monte Carlo iterations to perform
    :return: NumPy array, float, float tuple
    """
    expected_returns, log_returns = get_expected_returns(model_specs)
    var_covar_matrix = get_var_covar_matrix(log_returns.dropna())
    optimal_weights, portfolio_er, portfolio_sd = portfolio_weights_monte_carlo(log_returns,
                                                        expected_returns, var_covar_matrix, iterations=iterations)
    return optimal_weights, portfolio_er, portfolio_sd

# Notes regarding the UI
# "main" is what really needs to be called from a UI since the inputs are really long and it calls nearly everything in the project
# basically we are inputting what assets we want in the portfolio, what we want each asset to be potentially modelled by, and we should get
# recommended weights to hold for the next day
# regarding the input dictionary: if a key is "AAPL" the ticker set in the tuple "AAPL" maps to must always include "AAPL" or it will throw an error
# this means that if we are going to construct a portfolio with Apple stock we will always need Apple stock returns data
# to forecast its expected returns
# you can find more on what the really long inputs are doing in the functions get_expected_returns and get_data in model_fitting.py
# note the out of sample forecast will be for whatever day comes after your end date, so if end year is 2022 and end month is 2 the out of sample
# forecast would be for tomorrow which is what we generally want

# SAMPLE USE OF MAIN
#MODEL_SPECS = {"AAPL": ({"AAPL", "INTC", "MSFT"}, 3, "AAPL_log_return", 3, None, 2021, 5, 2022, 2, True),
              #"GME": ({"GME", "AMC", "BB"}, 3, "GME_log_return", 3, None, 2021, 5, 2022, 2, True),
             #"AMC": ({"AMC", "GME"}, 2, "AMC_log_return", 3, None, 2021, 5, 2022, 2, True),
               #"GOOGL": ({"GOOGL", "AAPL"}, 2, "GOOGL_log_return", 3, None, 2021, 5, 2022, 2, True)}
#print(main(MODEL_SPECS))
