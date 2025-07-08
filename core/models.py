from django.db import models
from django.contrib.auth.models import User


class UploadedPaper(models.Model):
    user=models.ForeignKey(User, on_delete=models.CASCADE)
    file=models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
# Create your models here.


class ExamAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    input_text = models.TextField()
    model_type = models.CharField(max_length=20, choices=[
        ('basic', 'Basic'),
        ('standard', 'Standard'),
        ('advanced', 'Advanced'),
    ])
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)