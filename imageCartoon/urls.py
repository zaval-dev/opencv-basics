from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='cartooning_index'),
    path('webcam/', views.open_webcam, name='open_webcam'),
    path('interactive/', views.rectangle, name='rectangle_camera'),
    path('median/', views.median_filter, name='median_filter'),
    path('bilateral/', views.bilateral_filter, name='bilateral_filter'),
    path('cartoon/', views.cartoonig_image, name='cartooning_image'),
]
