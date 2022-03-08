import pytrends
from pytrends.request import TrendReq
from pytrends import dailydata
import pandas as pd
import datetime
import dateutil
import yfinance_scraper
import statsmodels as sm
import matplotlib as plt
from dateutil import relativedelta

# this is the connection to Google in English and on EST (matches NYSE)
request = TrendReq(hl='en-US', tz=300)

# DATES AND TIMES FOR CONCATENATION
today_tuple = datetime.date.today()
today = today_tuple.strftime('%Y-%m-%d')

YTD = today[:4] + '-01-01 ' + today

prev_month_tuple = today_tuple + relativedelta.relativedelta(months=-1)
prev_month = prev_month_tuple.strftime('%Y-%m-%d')
M1 = prev_month + ' ' + today

six_month_tuple = today_tuple + relativedelta.relativedelta(months=-6)
six_months = six_month_tuple.strftime('%Y-%m-%d')
M6 = six_months + ' ' + today

# this is the set of valid timeframes that line up with yahoo timeframes
timeframes = {'Max': 'all', '5Y': 'today 5-y', '1Y': 'today 1-Y', 'YTD': YTD, '6M': M6, '1M': M1, '5D': 'now 7-d',
              '1D': None, '1H': 'now 1-H', '4H': 'now 4-H'}


def interest_table(keywords, time):
    '''
    Builds a Pandas Dataframe of interest in the keywords over the
    desired timeframe. Interest is in a range of 0 to 100, with peak interest
    being 100

    Inputs:
    keywords (list): list of phrases to query, EVEN WITH ONE TERM
    timeframe (string): desired timeframe in standard finance terms
    '''

    # Exception if not a standard finance timeframe
    if time not in timeframes:
        raise Exception(time, "Is not a valid timeframe")

    elif timeframes[time] == None:
        raise Exception("Google data is not compatible with this timeframe; try either 1H or 4H timeframes")

    else:
        request.build_payload(keywords, timeframe=timeframes[time])
        df = request.interest_over_time()
        df = df.drop('isPartial', axis=1)

    return df

def daily_interest_table(kw_lst, end_year, end_month, start_year = 2004, start_month=1):
    """
    Gets daily interest data for list of tickers
    If no start date requested, returns all data since the start of Google Trends (Jan 1 2004)
    Inputs:
        kw_lst: list of keywords
        start_year: (int)
        start_month: (int)
        end_year: (int)
        end_month: (int)
    Returns:
        DataFrame where each keyword has its own column
    """

    d = {}
    for word in kw_lst:
        df = dailydata.get_daily_data(word, start_year, start_month,
                                      end_year, end_month, wait_time=.10)
        interest = df[word]
        d[word] = interest
    output_df = pd.DataFrame(d).dropna()
    output_df.rename(columns={'date': 'Date'}, inplace=True)
    return output_df

def related_queries(keyword, time):
    '''
    Retrieves related queries to help the user ensure their search covers
    as much ground as possible

    Inputs:
    keyword (string): keyword to query
    timeframe (string): desired timeframe in standard finance terms

    Returns (dictionary): dictionary of related terms mapped to a rating
        from 0 to 100 of the related term
    '''

    request.build_payload([keyword], timeframe=timeframes[time])

    return request.related_queries()[keyword]['top']


def search_heat(ticker, keywords):
    '''

    Trains and tests an OLS model to predict Ticker log returns with keyword

    search density



    Inputs:

    ticker (string): stock ticker whose log returns we want to predict

    term_list (list of strings): list of search strings to test as predictors



    Returns: a plot of the residuals over the past year



    WARNING: as a security measure, Google forces a 60 second cooldown every

    100 terms, which will occur at least once per additional search term

    '''

    # This will grab the log returns dataframe

    ticker_set = set([ticker])

    dic = yfinance_scraper.get_yfinance_data(ticker_set)

    log_returns = yfinance_scraper.get_log_return_df(dic)

    # We will train on the 4 years preceeding last year and test on last year

    # This builds the complete search density dataset

    prev_five_years = datetime.date.today() + relativedelta.relativedelta(years=-5)

    search_density = daily_interest_table(keywords, today_tuple.year, today_tuple.month, prev_five_years.year,
                                          prev_five_years.month)

    one_year_ago = (datetime.date.today() + relativedelta.relativedelta(years=-1)).strftime('%Y-%m-%d')

    # we decompose the search density into training and testing periods

    train_search_density = search_density[search_density.index < one_year_ago]

    test_search_density = search_density[search_density.index >= one_year_ago]

    # merge the training and testing frames with the log returns frame

    train_frame = pd.merge(log_returns, train_search_density, left_index=True, right_index=True)

    test_frame = pd.merge(log_returns, test_search_density, left_index=True, right_index=True)

    model = sm.OLS(train_frame.iloc[:, 0], sm.add_constant(train_frame.iloc[:, 1:])).fit()

    predictions = model.predict(sm.add_constant(test_frame.iloc[:, 1:]))

    observations = test_frame.iloc[:, 0]

    residuals = observations.sub(predictions)

    residuals.plot(title="Residuals Over This Year")

    plt.show()

