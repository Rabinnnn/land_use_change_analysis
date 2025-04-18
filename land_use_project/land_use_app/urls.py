from django.urls import path
from . import views

urlpatterns = [
    path('', views.analyze_images, name='analyze_images'),
]
