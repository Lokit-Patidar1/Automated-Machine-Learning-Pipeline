from django.urls import path
from . import views

urlpatterns = [
	path('', views.upload_view, name='upload'),
	path('eda/', views.eda_view, name='eda'),
	path('train/', views.train_view, name='train'),
	path('download/', views.download_view, name='download'),
]
