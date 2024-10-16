from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='seam_index'),
    path('reduce-image/', views.reduce_image, name='reduce_image'),
    path('expand-image/', views.expand_image, name='expand_image'),
    path('vertical-waves/', views.vertical_waves, name='vertical_waves'),
    path('horizontal-waves/', views.horizontal_waves, name='horizontal_waves'),
    path('two-waves/', views.full_waves, name='full_waves'),
]
