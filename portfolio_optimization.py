import pandas as pd
import numpy as np
import random
import itertools
import datetime as dt
from itertools import combinations
from correlation_modelling import forecast_var_cov_matrix
from model_fitting import get_expected_returns
from itertools import permutations

expected_returns = np.matrix([[.1], [.1], [.04]]).T

df = pd.read_csv("steve_data.csv")
df = df.dropna()

mapping = {}
for i, col in enumerate([x for x in list(df.columns) if x != "Date"]):
    mapping[i] = col

#var_cov_matrix = np.matrix(forecast_var_cov_matrix(df))
#num_assets = var_cov_matrix.shape[1]

weights = [.5, .5]
weights = np.array(weights).reshape(2, -1)

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
    portfolio_er = float(expected_returns @ weights)
    sharpe = portfolio_er/sd
    return sharpe

def get_random_weights(num_assets):
    """
    Generates random portfolio weights that sum to 1
    :param num_assets: Number of weights to generate
    :return: Weights (NumPy array)
    """
    weights = np.random.random(num_assets) * 2 - 1
    weights /= weights.sum()
    weights = np.array(weights).reshape(num_assets, -1)
    return weights

def portfolio_weights_monte_carlo(df, expected_returns, var_cov_matrix, iterations=5000000):
    """
    Runs a Monte Carlo simulation to approximate optimal portfolio weights for an arbitrary number of assets
    :param df: DataFrame of log returns
    :param expected_returns: Array of expected returns
    :param var_cov_matrix: Forecasted variance-covariance matrix
    :param iterations: Number of random weights to try (default is 5 million, takes about 2 minutes to run)
    :return: Sharpe ratio, optimal weights (array) tuple
    """
    start_time = dt.datetime.now()
    num_assets = var_cov_matrix.shape[1]
    best_sharpe, best_weights = None, None
    for i in range(5000000):
        weights = get_random_weights(num_assets)
        sharpe = get_portfolio_sharpe(var_cov_matrix, expected_returns, weights)
        if best_sharpe is None:
            best_sharpe = sharpe
            best_weights = weights
        elif sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights
    end_time = dt.datetime.now()
    elapsed = end_time - start_time
    print(elapsed.seconds)
    return best_sharpe, best_weights

def main(model_specs):
    expected_returns, log_returns = get_expected_returns(model_specs)
    var_covar_matrix = get_var_covar_matrix(log_returns.dropna())
    optimal_weights = portfolio_weights_monte_carlo(log_returns, expected_returns, var_covar_matrix)
    return optimal_weights

#print(portfolio_weights_monte_carlo(df, expected_returns, get_var_covar_matrix(df)))

#MODEL_SPECS = {"AAPL": ({"AAPL", "INTC", "MSFT"}, 3, "AAPL_log_return", 3, None, 2021, 5, 2022, 2, True),
 #              "GME": ({"GME", "AMC", "BB"}, 3, "GME_log_return", 3, None, 2021, 5, 2022, 2, True),
  #             "AMC": ({"AMC", "GME"}, 2, "AMC_log_return", 3, None, 2021, 5, 2022, 2, True)}

#expected_returns, log_returns = get_expected_returns(MODEL_SPECS)

#var_covar_matrix = get_var_covar_matrix(log_returns.dropna())
#print(portfolio_weights_monte_carlo(log_returns, expected_returns, var_covar_matrix))

#print(get_portfolio_sharpe(var_cov_matrix, expected_returns, weights))

#z = list(range(10))
#c = combinations(z, 2)
