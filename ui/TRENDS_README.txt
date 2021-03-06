A GUIDE TO TRENDS.PY

- Trends uses a few libraries from Google to help the user think about how search trends may play out in their thesis
- Like any investment data software, trends contains helpful tools, but is NOT a magic 8-ball to predict the market
- It has 2 tasks, one of which the user will interact with directly:
    - Put daily google trends data into a dataframe to interact with the larger MAIN program
    - Allow the user to test theses regarding market sentiment and how an equity trades

REQUIRED MODULES/LIBRARIES:
- pytrends
- pandas
- numpy
- datetime
- dateutil
- statsmodels.api
- mathplotlib

FUNCTIONS FOR THE USER:

search_heat:
- To use: enter your desired ticker as a string (e.g. "JPM", use capital letters) and a list of search terms (also strings)
to try to find a correlation (e.g. ['interest rates','fed','federal reserve']). 
- search_heat tries to predict the log returns of an equity through the density of a set of search terms
on Google Trends. Generally speaking, this is meant to help value investment theses that want to argue that
an equity's price is not reflective of some key observation. While a low R^2 of a given search set is 
necessary to show that something is not being priced in, this tool will not compensate for a bad underlying thesis. 
Further, since google is very specific about the densities of search terms, we find it is best to run search_heat with 
MULTIPLE related terms to best encompass the concept you want to test. Search_heat is best used when considering long-only,
value-type theses that rely on extensive data spanning the past 5 years, rather than short term market sentiment. 


BACKGROUND FUNCTIONS:

Daily_interest_table and interest_table:
- These are simply retrievals of dataframes over the course of time to get search densities. They will employ
the global variables at the top of the file to help line themselves up with finance data retrieved elsewhere
in the program. These dataframes will serve as an additional component of the analysis done by the program. 

    
