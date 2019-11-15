from django.urls import path
from . import views

urlpatterns = [
    path('', views.get_topics, name='get_topics')
]
