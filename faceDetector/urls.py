from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='face_detector_index'),
    path('face-detector/', views.face_detector, name='face_app'),
    path('mask/', views.face_mask, name='batman_mask'),
    path('moustache/', views.moustache_mask, name='moustache_mask'),
    path('sunglasses/', views.sunglasses_mask, name='sunglasses_mask'),
    path('nose/', views.nose_mask, name='nose_mask'),
]
