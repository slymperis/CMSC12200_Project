import rpy2
import pandas as pd
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def forecast_var_cov_matrix(df):
    """
    Uses the rpy2 R API in Python to run R code that uses the rmgarch library
    to fit a DCC GARCH model that allows the variance-covariance matrix of assets to vary over time
    Returns the forecast of the one step out of sample variance-covariance matrix
    :param df: DataFrame of log returns
    :return: DataFrame of tomorrow's forecasted variance-covariance matrix
    """
    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    utils.install_packages(StrVector(("rmgarch",)))
    rmgarch = importr("rmgarch")
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df)
    robjects.r('''
        library(rmgarch)
        dcc_fcast <- function(returns, days_to_fcast){
                    returns <- returns[, ! names(returns) %in% "Date", drop = F]
                    univariate_spec <- ugarchspec(mean.model = list(armaOrder = c(0,0)),
                    variance.model = list(garchOrder = c(1,1),
                    variance.targeting = FALSE, 
                    model = "sGARCH"),
                    distribution.model = "std")
                    dims <- dim(returns)[2]
                    dcc_spec <- dccspec(uspec = multispec(replicate(dims, univariate_spec)),
                    dccOrder = c(1,1),
                    distribution = "mvt")
                    dcc_fit <- dccfit(dcc_spec, data=returns)
                    forecasts <- dccforecast(dcc_fit, n.ahead = days_to_fcast)
                    list(dcc_fit, as.data.frame(forecasts@mforecast$H))
                    }
    ''')
    dcc_fcast = robjects.globalenv['dcc_fcast']
    r_output = dcc_fcast(r_from_pd_df, 1)
    dcc_model_r = r_output[0]
    fcast_var_cov_r = r_output[1]
    with localconverter(robjects.default_converter + pandas2ri.converter):
        fcast_var_cov = robjects.conversion.rpy2py(fcast_var_cov_r)
    return fcast_var_cov
