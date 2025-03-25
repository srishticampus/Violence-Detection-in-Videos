from django.urls import path,include
from . import views

urlpatterns = [
    path('',views.index,name='index'),

    path('signup/', views.user_signup, name='user_signup'),
    path('signin/', views.user_signin, name='user_signin'),
    path('logout/', views.user_logout, name='user_logout'),
    path('con',views.contact,name='contact'),
    path('ab',views.about,name='about'),
    path('home',views.home,name='home'),
    path('upload/', views.upload_image_view, name='upload_image'),
    path('success/', views.upload_success, name='image_upload_success'),
    path('upload-video/', views.upload_video_view, name='upload_video'),
    path('video-success/', views.video_upload_success, name='video_upload_success'),




]
