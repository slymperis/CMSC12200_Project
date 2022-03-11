import json
import traceback
import sys
import csv
import os
import numpy as np

from trends import search_heat

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from django.shortcuts import render
from django.http import HttpResponse

from portfolio_optimization import main
from django.views.generic import TemplateView, DetailView
from django.views import View

from PIL import Image
import base64
from io import BytesIO
        
'''COLUMN_NAMES = dict(
    b_weights = 'Best Weights'
    expected_return = 'Optimal Portfolio Expected Return'
    expected_SD = 'Optimal Portfolio Standard Deviation'
)'''

err_string = """
             An exception was thrown in find_courses:
                         <pre>{}
             {}</pre>
             """
no_keys = "form does not contain key words"

default_img = np.array([[[255, 255, 255]]], dtype=np.uint8)

# class AllPageView(DetailView):

#     def get(self, request):

def get_uri_from_rgbarray(rgb_array):
    img = Image.fromarray(rgb_array, 'RGB')  
    data = BytesIO()
    img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 
        

class TrendsPageView(DetailView): 
    template_name = "trends.html"
    def get(self, request):
        context = {}
        keywords_str = []

        assert request.method == 'GET'
        form = TrendsForm(request.GET)
        if form.is_valid():
            assert 'key_words' in form.cleaned_data, no_keys
            for word in form.cleaned_data['key_words'].split(): 
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

class PortfolioPageView(DetailView): 
    template_name = "index.html"
    def get(self, request):
        context = {}
        res = None 
        if request.method == 'GET':
            print("If request.method == GET")
            form = SearchForm(request.GET)
            if form.is_valid():
                args = {}
                args_val = []
                args_val.append({regress_ticker})
                args_val.append(num_lags, stock_query + "_log_return", num_models)
                if key_words:
                    args_val.append(key_words)
                else:
                    args_val.append(None)
                args_val.append(start_time_year, start_time_month, end_time_year, end_time_month)
                if analyst_recs:
                    args_val.append(True)
                else:
                    args_val.append(False)
            
                args[key] = tuple(args_val)

                try:
                    res = main(args)

                except Exception as e:
                    print('Exception caught')
                    bt = traceback.format_exception(*sys.exc_info()[:3])
                    context['err'] = """
                    An exception was thrown in find_courses:
                    <pre>{}
    {}</pre>
                    """.format(e, '\n'.join(bt))

                    res = None

        else:
            form = SearchForm()
            
        if res is None:
            context['result'] = None
        elif isinstance(res, str):
            context['result'] = None
            context['err'] = res
            result = None
        elif not _valid_result(res):
            context['result'] = None
            context['err'] = ('Return of portfolio_optimization has the wrong data type.')
        else:
            weights, result_er, result_op = res

            context['weights'] = weights
            context['result_er'] = result_er 
            context['result_op'] = result_op
            context['columns'] = [COLUMN_NAMES.get(col, col) for col in columns]    #don't really know what this does


        context['form'] = form

        return render(request, 'index.html', context)


class TrendsForm(forms.Form):
    stock_ticker = forms.CharField(
        label = 'Stock Ticker:',
        help_text = 'e.g. AAPL',
        required = True)
    key_words = forms.CharField(
        label = 'Key Words to Query for Google Trends:',
        help_text = 'e.g. iPhone',
        required = False)
    

class SearchForm(forms.Form):
    stock_query = forms.CharField(
        label = 'Stock Ticker:',
        help_text = 'e.g. AAPL',
        required = True)
    regress_ticker = forms.CharField(
        label = 'Ticker Data to Regress On:',
        help_text = 'e.g. AAPL, INTC, MSFT (separated by commas)',
        required = True)
    num_models = forms.CharField(
        label = 'Number of Models to Evaluate:',
        help_text = 'e.g. 3',
        required = True)
    num_lags = forms.CharField(
        label = 'Number of Lags',
        help_text = 'e.g. 3',
        required = True)
    key_words = forms.CharField(
        label = 'Key Words to Query for Google Trends:',
        help_text = 'e.g. iPhone (separated by commas); leave blank if None',
        required = False)
    start_time_month = forms.CharField(
        label = 'Start Time (Month)',
        help_text = 'e.g. 3 (March)',
        required = True)
    start_time_year = forms.CharField(
        label = 'Start Time (Year)',
        help_text = 'e.g. 2021',
        required = True)
    end_time_month = forms.CharField(
        label = 'End Time (Month)',
        help_text = 'e.g. 5 (May)',
        required = True)
    end_time_year = forms.CharField(
        label = 'End Time (Year)',
        help_text = 'e.g. 2022',
        required = True)
    analyst_recs = forms.BooleanField(label = 'Include Analyst Recommendations',
        required = False)
