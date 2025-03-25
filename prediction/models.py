from django.db import models

# Create your models here.
from django.db import models

class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} uploaded at {self.uploaded_at}"



class VideoUpload(models.Model):
    video = models.FileField(upload_to='uploaded_videos/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Video {self.id} uploaded at {self.uploaded_at}"
