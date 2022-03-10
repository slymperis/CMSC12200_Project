from django.urls import path

from . import views
from .views import TrendsPageView, PortfolioPageView

urlpatterns = [
    path('', PortfolioPageView.as_view(), name='index'),
    path("trends/", TrendsPageView.as_view(), name="trends"),
    #path('', HomePageView.as_view(), name="home"),
]