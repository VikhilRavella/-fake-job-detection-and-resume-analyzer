# main_app/api_urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('detect/text/', views.manual_predict_view, name='predict_text'),
    # ... any other API urls
]