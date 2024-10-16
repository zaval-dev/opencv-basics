from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='reality_index'),
    path('pose/', views.pose_estimator, name='reality_pose'),
    path('piramid/', views.piramid, name='reality_piramid'),
    path('tetraed/', views.tetraed, name='reality_tetraed'),
]
