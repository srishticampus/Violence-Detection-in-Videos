from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages, auth
from .forms import ImageUploadForm
from .models import ImageUpload
from .forms import VideoUploadForm
from .models import VideoUpload

import os
import cv2
import numpy as np
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings
import gdown

# Prediction code section starting



# Define class labels
class_labels = ['Fight', 'No Fight']
IMG_SIZE = (299, 299)


def upload_image(request):
    """Image Prediction View"""
    prediction_label = None

    if request.method == 'POST' and request.FILES['image']:
        # Save uploaded image
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_image.name, uploaded_image)
        image_path = fs.url(image_path)

        # Predict the image
        img_path = os.path.join(settings.MEDIA_ROOT, uploaded_image.name)
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        prediction_label = class_labels[np.argmax(prediction)]

    return render(request, 'upload_image.html', {'prediction_label': prediction_label})


def process_frame(frame):
    """Preprocess video frames for prediction."""
    frame_resized = cv2.resize(frame, IMG_SIZE)
    frame_normalized = frame_resized / 255.0
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    return frame_expanded


def predict_video(video_path, output_path):
    """Predict the entire video and save with labels."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Set up video writer (use 'mp4v' for .mp4 files)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = output_path if output_path.endswith('.mp4') else f"{output_path}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict frame
        processed_frame = process_frame(frame)
        prediction = model.predict(processed_frame)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]

        # Draw prediction on frame
        label = f"{class_labels[predicted_class]} ({confidence * 100:.2f}%)"
        color = (0, 0, 255) if predicted_class == 0 else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (width, height), color, 10)
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Write frame to output
        out.write(frame)

    cap.release()
    out.release()

    print(f"Video saved at: {output_path}")



from django.shortcuts import render
from django.conf import settings
import os

def upload_video(request):
    """Video Prediction View"""
    predicted_video_url = None

    if request.method == 'POST' and request.FILES.get('video'):
        # Save uploaded video
        uploaded_video = request.FILES['video']
        video_path = os.path.join(settings.MEDIA_ROOT, uploaded_video.name)

        # Save video to media directory
        with open(video_path, 'wb+') as destination:
            for chunk in uploaded_video.chunks():
                destination.write(chunk)

        # ✅ Print saved video path to check its location
        print("Saved video path:", video_path)

        # Generate output video path
        output_name = f"predicted_{uploaded_video.name}"
        output_path = os.path.join(settings.MEDIA_ROOT, output_name)

        # ✅ Print output video path (where the predicted video is saved)
        print("Predicted video path:", output_path)

        # Run prediction
        predict_video(video_path, output_path)

        # Generate URL to access processed video
        predicted_video_url = f"{settings.MEDIA_URL}{output_name}"
        print(f"Predicted Video URL: {predicted_video_url}")

    return render(request, 'upload_video.html', {'predicted_video_url': predicted_video_url})



# Prediction code section ending

# Load the trained model once when the server starts
model_path = os.path.join(settings.BASE_DIR, 'fight_detection_model.h5')
def download_model_if_needed():
    """Check if the model exists, if not, download it from Google Drive."""
    if not os.path.exists(model_path):
        print("Model not found. Downloading...")
        drive_link = 'https://drive.google.com/uc?id=1FYDBtyrXItx3ZIDmLCeR_C6TrcPY-Ye2'
        gdown.download(drive_link, model_path, quiet=False)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# Download model if not present
download_model_if_needed()

# Load the trained model once when the server starts
model = load_model(model_path)





def articles(request):
    return render(request, 'articles.html')

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



# def predict_video(request):
#     return render(request, 'upload_video.html')

def predict_image(request):
    return render(request, 'upload_image.html')

# def upload_video(request):
#     if request.method == "POST":
#         # Handle video upload and prediction logic here
#         pass
#     return render(request, 'video_prediction.html')

# def upload_image(request):
#     if request.method == "POST":
#         # Handle image upload and prediction logic here
#         pass
#     return render(request, 'image_prediction.html')
