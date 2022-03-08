import json
import traceback
import sys
import csv
import os

from functools import reduce
from operator import and_

from django.shortcuts import render
from django import forms

from django.shortcuts import render
from django.http import HttpResponse

#from portfolio_optimization import main

class SearchForm(forms.Form):
    stock_query = forms.CharField(
        label = 'Stock Ticker:', 
        help_text = 'e.g. AAPL', 
        required = True)
    key_words = forms.CharField(
        label = 'Key Words to Query:',
        help_text = 'e.g. iPhone',
        required = False)

def index(request):
    context = {}
    res = None 
    if request.method == 'GET':
        form = SearchForm(request.GET)
    
    return render(request, 'index.html', context)
    #return HttpResponse("You have arrived at the returns index for JAWS")

    