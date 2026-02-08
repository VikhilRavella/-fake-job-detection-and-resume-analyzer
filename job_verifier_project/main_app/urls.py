from django.urls import path
from . import views

urlpatterns = [
    # Homepage
    

    # ❌ ML / Heavy routes DISABLED for Render free tier
    # path('predict/', views.manual_predict_view, name='predict'),
    # path('agent/', views.agent_view, name='agent'),

    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
]
