from django.urls import path
from . import views

urlpatterns = [
    # URL for your homepage (e.g., job_verifier_project.com/)
    path('', views.home_view, name='home'),

    # URL for the manual prediction page
    path('predict/', views.manual_predict_view, name='predict'),
    #path('predict/', views.predict_view, name='predict'),
    # URL for the AI Agent page
    path('agent/', views.agent_view, name='agent'),

    # URLs for your other pages
    path('about/', views.about_view, name='about'),
    path('contact/', views.contact_view, name='contact'),
]