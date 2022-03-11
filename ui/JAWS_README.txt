JAWS PROJECT README

REQUIREMENTS (please read):
Pandas 1.3.5- NOT PANDAS 1.4.1 as the Yahoo finance API wrapper has not yet been updated to work with the latest version of Pandas
R to run correlation modelling not found in any Python packages
rmgarch package- should be installed on department machines after talking to tech staff, if there is an issue
with rmgarch refer to tech staff as they said they would update it

WHAT IT DOES:

The program pulls data from several sources: log returns of itself and other user-provided tickers, Google trends search interest, and analyst recommendations
from Yahoo finance that are automatically scored. It then creates as many lags as the user wants to regress on and splits the data into training and testing sets
and fits optimal linear, random forest, and gradient booster models. It ultimately selects the model deemed most appropriate and uses it to forecast tomorrow's
expected return matrix. It then uses a Dynamic Conditional Correlations model to forecast tomorrow's variance covariance matrix. Finally, it selects optimal weights
on each asset to maximize the Sharpe ratio defined as expected return divided by portfolio variance subject to the conditions that the weights sum to one and that
no single weight is greater than 1 or less than -1 (negative implying short positions). As the number of weights increases, this becomes impossible to solve
analytically and thus a Monte Carlo simulation is used. The optimal weights, portfolio expected return, and portfolio expected standard deviation is then displayed.

UI:

Stock Ticker: Each ticker in this field must be a valid Yahoo finance ticker. These are the tickers that will be part of the final optimal portfolio, which means their
expected returns will be modelled.

Ticker Data to Regress On: Seperate tickers to use in the same regression by commas, tickers to use in different regressions by semicolons. NOTE: if you are trying
to model AAPL's expected returns, you MUST include AAPL in the tickers to regress on. For example, if the Stock Ticker field includes "AAPL, GME" and the 
Ticker Data to Regress On field includes "AAPL, INTC, MSFT; GME, AMC" the program will construct a portfolio of AAPL and GME stock where AAPL is regressed on
AAPL, INTC, and MSFT and GME is regressed on GME and AMC.

Number of Models to Evaluate: The number of good linear models from the training set to test in the validation set. This only applies to linear models.

Number of Lags: The number of lags to create and fit models on. For instance, if lags is set to 2 then the model fitting procedure will be able to use
up to 2 lags. It may not necessarily use all lags, but it will try to use all lags and evaluate its performance.

Key Words to Query for Google Trends: The words to associate with each ticker. Seperated by commas if in the same regression, seperated by semicolons if in different
regressions.

Start Time (Month): The month to start gathering data for. Older start dates will take much more time to run as the program needs to query the Google Trend's API.

Start Time (Year): The year to start gathering data for. Note: start date must be more recent than Jan 1, 2004 as no earlier Google Trend's data exists.