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


    
class ExamPaper(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    subject = models.CharField(max_length=100)
    file = models.FileField(upload_to='exam_papers/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-uploaded_at']

class SolutionView(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    paper = models.ForeignKey(ExamPaper, on_delete=models.CASCADE)
    viewed_at = models.DateTimeField(auto_now_add=True)
    was_helpful = models.BooleanField(null=True, blank=True)

class UserActivity(models.Model):
    ACTIVITY_TYPES = (
        ('upload', 'Paper Uploaded'),
        ('view', 'Solution Viewed'),
        ('feedback', 'Feedback Provided'),
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    paper = models.ForeignKey(ExamPaper, on_delete=models.SET_NULL, null=True, blank=True)
    details = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=200)

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    sender = models.CharField(max_length=10)  # 'user' or 'bot'
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    knowledge_level = models.CharField(max_length=20, default='intermediate')