from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='shapes_index'),
    path('triangle/', views.contour_matching_with_triangle, name='shapes_triangle'),
    path('contours/', views.contours, name='shapes_contours'),
    path('circle/', views.contour_matching_with_circle, name='shapes_circle'),
    path('convex/', views.convexity, name='shapes_convex'),
    path('watershed/', views.watershed, name='shapes_watershed'),
]
