from django.urls import path
from . import views

urlpatterns = [
    path('',views.index, name='tracking_index'),
    path('gmg/',views.bg_sustraction_gmg, name='tracking_gmg'),
    path('mog/',views.bg_sustraction_mog, name='tracking_mog'),
    path('blue/',views.blue_detection, name='tracking_blue'),
    path('tracker/',views.tracker, name='tracking_tracker'),
    path('green/',views.green_detection, name='tracking_green'),
]
