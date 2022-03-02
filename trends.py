# TRENDS TO DATAFRAME PROGRAM -WILL GRIFFIN

# key imports
import pytrends
from pytrends.request import TrendReq
import pandas as pd
import datetime
import dateutil
from dateutil import relativedelta

# this is the connection to Google in English and on EST (matches NYSE)
request = TrendReq(hl='en-US', tz=300)

# DATES AND TIMES FOR CONCATENATION
today_tuple = datetime.date.today()
today = today_tuple.strftime('%Y-%m-%d')

YTD = today[:4] + '-01-01 ' + today

prev_month_tuple = today_tuple + relativedelta.relativedelta(months = -1)
prev_month = prev_month_tuple.strftime('%Y-%m-%d')
M1 = prev_month + ' ' + today

six_month_tuple = today_tuple + relativedelta.relativedelta(months = -6)
six_months = six_month_tuple.strftime('%Y-%m-%d')
M6 = six_months + ' ' + today

# this is the set of valid timeframes that line up with yahoo timeframes
timeframes = {'Max': 'all', '5Y': 'today 5-y', '1Y':'today 1-Y', 'YTD': YTD, '6M': M6, '1M': M1, '5D': 'now 7-d', '1D': None, '1H': 'now 1-H', '4H': 'now 4-H'}

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
        
    # temporary exception, should eventually go away as google and Yfinance are resolved
    elif timeframes[time] == None:
        raise Exception("Google data is not compatible with this timeframe; try either 1H or 4H timeframes")
        
    else:
        request.build_payload(keywords, timeframe = timeframes[time])
        df = request.interest_over_time()
        df = df.drop('isPartial', axis = 1)
        
    return df

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
    
    request.build_payload([keyword], timeframe = timeframes[time])
    
    return request.related_queries()[keyword]['top']