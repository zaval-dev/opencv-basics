from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='cartooning_index'),
    path('interactive/', views.rectangle, name='rectangle_camera'),
]
