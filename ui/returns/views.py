import json
import traceback
import sys
import csv
import os
import numpy as np
import datetime as dt

from trends import search_heat
from portfolio_optimization import main

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from django.shortcuts import render
from django.http import HttpResponse

from django.views.generic import DetailView
from django.views import View

from PIL import Image
import base64
from io import BytesIO
        
err_string = """
             An exception was thrown in find_courses:
                         <pre>{}
             {}</pre>
             """
no_keys = "form does not contain key words"

default_img = np.array([[[255, 255, 255]]], dtype=np.uint8)

def get_uri_from_rgbarray(rgb_array):
    '''
        Helper function used in the get method of TrendsPageView; aids in 
        displaying an image to the user depending on their inputs. 

        Input: rgb_array
        Output: uri
    '''
    img = Image.fromarray(rgb_array, 'RGB')  
    data = BytesIO()
    img.save(data, "JPEG")
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
        
#View Class for JAWS Trends, which calls search_heat in trends.py
class TrendsPageView(DetailView): 
    template_name = "trends.html"
    def get(self, request):
        context = {}
        keywords_str = []

        assert request.method == 'GET'
        form = TrendsForm(request.GET)
        if form.is_valid():
            assert 'key_words' in form.cleaned_data, no_keys
            for word in form.cleaned_data['key_words'].split(" "): 
                keywords_str.append(word)
            stock_ticker = form.cleaned_data['stock_ticker']

            try:
                image, summary = search_heat(stock_ticker, keywords_str)
            except Exception as e:
                print('Exception caught')
                bt = traceback.format_exception(*sys.exc_info()[:3])
                print(err_string.format(e, '\n'.join(bt)))
                image, summary = default_img, ""
        
            try:
                context['image'] = get_uri_from_rgbarray(image)
            except Exception as e:
                print('Exception caught')
                bt = traceback.format_exception(*sys.exc_info()[:3])
                print(err_string.format(e, '\n'.join(bt)))
                context['image'] = None
                context['err'] = ('Regression plot RGB array has wrong data type.')

            context['summary'] = summary
            if not isinstance(context['summary'], str):
                context['summary'] = ""
                context['err'] = ('Summary has wrong data type.')

        else:
            context['image'] = get_uri_from_rgbarray(default_img)
            context['summary'] = ""

        context['form'] = form

        return render(request, 'trends.html', context)

#View Class for JAWS Portfolio Optimization, which calls main
class PortfolioPageView(DetailView): 
    template_name = "index.html"
    def get(self, request):
        context = {}
        res = None 
        assert request.method == 'GET'

        form = SearchForm(request.GET)
        if form.is_valid():
            args_dict = {}
            
            keys = form.cleaned_data['stock_query'].replace(' ', '').split(',')

            rec = form.cleaned_data['analyst_recs']

            stock_regress = form.cleaned_data['regress_ticker'].replace(' ', '')
            stock_regress = [set(s.split(',')) for s in stock_regress.split(";")]
            
            key_words = form.cleaned_data['key_words'].replace(' ', '')
            key_words = [s.split(',') for s in key_words.split(';')]
            key_words = [[s for s in word_lst if s != ''] for word_lst in key_words]
            key_words = [word_lst if word_lst else None for word_lst in key_words]

            process_nums = lambda field: [int(s) for s in form.cleaned_data[field].split(',')]
            models = process_nums('num_models')
            lags = process_nums('num_lags')
            start_months = process_nums('start_time_month')
            start_years = process_nums('start_time_year')
            end_months = [dt.datetime.now().month for x in start_months]
            end_years = [dt.datetime.now().year for x in start_years]

            tup_iter = zip(stock_regress, lags, models, key_words, start_years, 
                           start_months, end_years, end_months)
            arg_dict = {key: (*tup[:2], f'{key}_log_return', *tup[2:], rec) 
                        for key, tup in zip(keys, tup_iter)}
            
            weights, er, sd = main(arg_dict)
            context['output'] = str((weights, er, sd))
        else:
            context['output'] = ""

        context['form'] = form

        return render(request, 'index.html', context)

#Form used for JAWS Trends Page 
class TrendsForm(forms.Form):
    stock_ticker = forms.CharField(
        label = 'Stock Ticker:',
        help_text = 'e.g. AAPL',
        required = True)
    key_words = forms.CharField(
        label = 'Key Words to Query for Google Trends:',
        help_text = 'e.g. iPhone iPad (separated by spaces)',
        required = False)
    
#Form used for JAWS Portfolio Optimization Page
class SearchForm(forms.Form):
    stock_query = forms.CharField(
        label = 'Stock Ticker:',
        help_text = 'e.g. AAPL, GME (separated by commas)',
        required = True)
    regress_ticker = forms.CharField(
        label = 'Ticker Data to Regress On:',
        help_text = 'e.g. AAPL, INTC, MSFT; GME, MSFT (separate regress tickers for each individual asset with semicolons) \
                    NOTE: You must include the main ticker in its own tickers to regress on as shown in the examples',
        required = True)
    num_models = forms.CharField(
        label = 'Number of Models to Evaluate:',
        help_text = 'e.g. 3, 4 (separate by commas)',
        required = True)
    num_lags = forms.CharField(
        label = 'Number of Lags',
        help_text = 'e.g. 3, 3 (separate by commas)',
        required = True)
    key_words = forms.CharField(
        label = 'Key Words to Query for Google Trends:',
        help_text = 'e.g. iPhone; Games (separate a list of key words for each individual asset with semicolons)',
        required = False)
    start_time_month = forms.CharField(
        label = 'Start Time (Month)',
        help_text = 'e.g. 3, 4 (March, April) (separate by commas)',
        required = True)
    start_time_year = forms.CharField(
        label = 'Start Time (Year)',
        help_text = 'e.g. 2020, 2021 (separate by commas)',
        required = True)
    analyst_recs = forms.BooleanField(label = 'Include Analyst Recommendations',
        required = False)
