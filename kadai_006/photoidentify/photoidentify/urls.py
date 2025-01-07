from django.urls import path
from django.contrib import admin
from prediction.views import predict

urlpatterns = [
     path('', predict, name='predict'),
]
