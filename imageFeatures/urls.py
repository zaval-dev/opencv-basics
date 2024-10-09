from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='features_index'),
    path('corners/', views.detect_corners, name='detect_corners'),
    path('main-corners/', views.detect_main_corners, name='detect_main_corners'),
    path('SIFT-algorithm/', views.sift_algorithm, name='sift_algorithm'),
    path('FAST-algorithm/', views.fast_algorithm, name='fast_algorithm'),
    path('ORB-algorithm/', views.orb_algorithm, name='orb_algorithm'),
]
