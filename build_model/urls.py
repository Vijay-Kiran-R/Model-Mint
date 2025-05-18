from django.urls import path
from . import views

urlpatterns = [
    path('', views.build_model_info, name='build_model'),
    path('upload/', views.upload_view, name='upload'),
    path('profiling/', views.profiling_view, name='profiling'),
    path('preprocessing/', views.preprocessing_view, name='preprocessing'),
    path('model/', views.model_view, name='model'),
    path("upload-to-blockchain/", views.upload_to_blockchain, name="upload_to_blockchain"),
    path('predict/', views.predict_view, name='predict'),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("fetch-metadata/", views.fetch_model_metadata, name="fetch_model_metadata"),
    path('download/', views.download_view, name='download'),
]