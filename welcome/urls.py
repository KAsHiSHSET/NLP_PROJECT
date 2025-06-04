from . import views
from django.urls import path

urlpatterns = [
    path('',views.Welcome,name='Welcome'),
    path('index/',views.Index,name='Index'),
    path('index/result/',views.Result,name='Result'),
]