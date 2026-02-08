# main_app/api_urls.py

from django.urls import path
# from . import views   # ❌ disabled – ML view causes OOM on Render free tier

urlpatterns = [
    # path('detect/text/', views.manual_predict_view, name='predict_text'),
    # ML API temporarily disabled to allow site to load
]
