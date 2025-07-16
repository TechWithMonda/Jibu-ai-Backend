from django.db import models
from django.contrib.auth.models import User
import uuid

# -------------------------------
# Document Upload and Plagiarism
# -------------------------------

class Document(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200)
    content = models.TextField()
    file = models.FileField(upload_to='documents/', null=True, blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.title

class PlagiarismReport(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    overall_similarity = models.FloatField()
    total_matches = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report for {self.document.title} - {self.overall_similarity}%"

class SimilarityMatch(models.Model):
    report = models.ForeignKey(PlagiarismReport, on_delete=models.CASCADE, related_name='matches')
    source_document = models.ForeignKey(Document, on_delete=models.CASCADE)
    similarity_score = models.FloatField()
    matched_text = models.TextField()
    source_text = models.TextField()
    start_position = models.IntegerField()
    end_position = models.IntegerField()

# -------------------------------
# Exam Analysis and Papers
# -------------------------------

class UploadedPaper(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

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

    def __str__(self):
        return f"{self.name} - {self.subject}"

class SolutionView(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    paper = models.ForeignKey(ExamPaper, on_delete=models.CASCADE)
    viewed_at = models.DateTimeField(auto_now_add=True)
    was_helpful = models.BooleanField(null=True, blank=True)

# -------------------------------
# User Activity Logging
# -------------------------------

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

# -------------------------------
# Chat / AI Conversation System
# -------------------------------

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=200)

    def __str__(self):
        return self.title

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    sender = models.CharField(max_length=10)  # 'user' or 'bot'
    content = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    knowledge_level = models.CharField(max_length=20, default='intermediate')

# -------------------------------
# Document Upload (General)
# -------------------------------

class UploadedDocument(models.Model):
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

# -------------------------------
# Premium Users and Payments
# -------------------------------

class PremiumUser(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    email = models.EmailField(unique=True)
    plan = models.CharField(max_length=50)
    reference = models.CharField(max_length=100)
    activated_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.email

class Payment(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    reference = models.CharField(max_length=100, unique=True)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='NGN')
    status = models.CharField(max_length=20)
    paid_at = models.DateTimeField(null=True, blank=True)
    gateway_response = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-paid_at']

    def __str__(self):
        return f"{self.user.username} - {self.status} - {self.reference}"

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    is_paid = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} - {'PAID' if self.is_paid else 'FREE'}"
