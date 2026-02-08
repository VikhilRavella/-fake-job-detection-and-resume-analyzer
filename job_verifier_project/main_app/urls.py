from django.urls import path
from . import views

urlpatterns = [
    # Homepage
    path('', views.home_view, name='home'),

    # ❌ ML / Heavy routes DISABLED for Render free tier
    # path('predict/', views.manual_predict_view, name='predict'),
    # path('agent/', views.agent_view, name='agent'),

    # Static HTML pages (SAFE)
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
]
