from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name="detect_objetcs_index"),
    path('dense/', views.dense_images, name="dense_images"),
    path('sift/', views.sift_images, name="sift_images"),
    path('dogs/', views.dogs, name="dog_breeds"),
    path('dog-or-cat/', views.animals, name="dog_cat"),
]
