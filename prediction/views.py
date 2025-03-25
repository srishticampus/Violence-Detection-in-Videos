from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages, auth
from .forms import ImageUploadForm
from .models import ImageUpload
from .forms import VideoUploadForm
from .models import VideoUpload





# Create your views here.
def index(request):
    return render(request,'index.html')


def contact (request):
    return render(request,'contact.html')



def about (request):
    return render(request,'about.html')


def home (request):
    return render(request,'about.html')



def user_signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']

        if password == confirm_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists')
                return redirect('user_signup')
            elif User.objects.filter(email=email).exists():
                messages.error(request, 'Email already used')
                return redirect('user_signup')
            else:
                user = User.objects.create_user(username=username, email=email, password=password)
                user.save()
                messages.success(request, 'Account created successfully. Please sign in.')
                return redirect('user_signin')
        else:
            messages.error(request, 'Passwords do not match')
            return redirect('user_signup')
    return render(request, 'signup.html')

def user_signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('index')  # Change to your homepage
        else:
            messages.error(request, 'Invalid credentials')
            return redirect('user_signin')
    return render(request, 'signin.html')

def user_logout(request):
    auth.logout(request)
    return redirect('user_signin')




def upload_image_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('image_upload_success')  # You can create a success page or redirect back to the form
    else:
        form = ImageUploadForm()
    
    images = ImageUpload.objects.all().order_by('-uploaded_at')  # Optional: Show previously uploaded images
    return render(request, 'upload_image.html', {'form': form, 'images': images})

def upload_success(request):
    return render(request, 'upload_success.html')



def upload_video_view(request):
    if request.method == 'POST':
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_upload_success')  # You can make a simple success page
    else:
        form = VideoUploadForm()

    videos = VideoUpload.objects.all().order_by('-uploaded_at')  # Optional: display all uploaded videos
    return render(request, 'upload_video.html', {'form': form, 'videos': videos})

def video_upload_success(request):
    return render(request, 'video_upload_success.html')


