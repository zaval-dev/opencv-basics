from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='transformation_index'),
    path('convertImage/', views.first_app, name='convert_image'),
    path('colorSpaces/', views.second_app, name='color_spaces'),
    path('rotate/', views.third_app, name='rotate_image'),
    path('resize/', views.quarter_app, name='resize_image'),
    path('affine/', views.fifth_app, name='deform_image'),
]