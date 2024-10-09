from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='filters_index'),
    path('blur/', views.blur_filter, name='blur_filter'),
    path('sharpen/', views.sharp_filter, name='sharpen_filter'),
    path('vignette/', views.vignette_filter, name='vignette_filter'),
    path('erosion-dilation/', views.erosion_dilation_filter, name='erosion_dilation_filter'),
    path('contrast/', views.enhacing_contrast_filter, name='contrast_filter'),
]