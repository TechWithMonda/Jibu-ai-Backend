import logging
from rest_framework import serializers
from django.contrib.auth import get_user_model, authenticate
from django.contrib.auth.models import User as User
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework.exceptions import ValidationError

from .models import (
    UploadedPaper, ExamPaper, SolutionView,
    UserActivity, Conversation, Message, Document, PlagiarismReport, SimilarityMatch)

logger = logging.getLogger(__name__)
User = get_user_model()

class TutorRequestSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)
    knowledge_level = serializers.CharField(required=False, default='intermediate')
    conversation_id = serializers.IntegerField(required=False, allow_null=True)  # Changed this line
    action = serializers.CharField(required=False)

    def validate(self, data):
        if len(data['message']) > 1000:
            raise ValidationError("Message too long (max 1000 characters)")
            
        if data.get('action') and data['action'] not in ['related', 'simplify', 'example', 'practice']:
            raise ValidationError("Invalid action specified")
            
        return data


class TutorResponseSerializer(serializers.Serializer):
    response = serializers.CharField(required=True)
    conversation_id = serializers.IntegerField(required=True)
    status = serializers.CharField(default='success')
    
    def validate_response(self, value):
        if not value or not value.strip():
            raise ValidationError("Response cannot be empty")
        return value

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'sender', 'content', 'timestamp', 'knowledge_level']
        read_only_fields = ['id', 'timestamp']

    def validate_sender(self, value):
        if value not in ['user', 'bot']:
            raise ValidationError("Sender must be either 'user' or 'bot'")
        return value

class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    
    class Meta:
        model = Conversation
        fields = ['id', 'title', 'created_at', 'user', 'messages']
        read_only_fields = ['id', 'created_at', 'user']
        
    def validate_title(self, value):
        if len(value) > 100:
            raise ValidationError("Title too long (max 100 characters)")
        return value

class ExamPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = ExamPaper
        fields = ['id', 'name', 'subject', 'uploaded_at']

class UserActivitySerializer(serializers.ModelSerializer):
    icon = serializers.SerializerMethodField()
    color = serializers.SerializerMethodField()

    class Meta:
        model = UserActivity
        fields = ['id', 'activity_type', 'details', 'created_at', 'icon', 'color']

    def get_icon(self, obj):
        icons = {
            'upload': "M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12",
            'view': "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z",
            'feedback': "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        }
        return icons.get(obj.activity_type, "")

    def get_color(self, obj):
        colors = {
            'upload': "blue",
            'view': "green",
            'feedback': "purple"
        }
        return colors.get(obj.activity_type, "gray")

class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        data = super().validate(attrs)
        
        # Get the user object
        user = self.user
        
        # Add comprehensive user data to the response
        data['user'] = {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'first_name': user.first_name,
            'last_name': user.last_name,
            'is_staff': user.is_staff,
            'is_active': user.is_active,
            'date_joined': user.date_joined.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return data

class RegisterSerializer(serializers.ModelSerializer):
    full_name = serializers.CharField(write_only=True)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)
    confirm_password = serializers.CharField(write_only=True)
    terms_agreed = serializers.BooleanField(write_only=True)

    class Meta:
        model = User
        fields = ['full_name', 'email', 'password', 'confirm_password', 'terms_agreed']

    def validate(self, data):
        if data['password'] != data['confirm_password']:
            raise serializers.ValidationError("Passwords do not match.")
        if not data.get('terms_agreed'):
            raise serializers.ValidationError("You must agree to the Terms of Service.")
        return data

    def create(self, validated_data):
        return User.objects.create_user(
            username=validated_data['email'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['full_name']
        )

class UploadedPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedPaper
        fields = ['id', 'file', 'uploaded_at']



class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'content', 'file', 'created_at', 'updated_at']

class SimilarityMatchSerializer(serializers.ModelSerializer):
    source_document = DocumentSerializer(read_only=True)
    
    class Meta:
        model = SimilarityMatch
        fields = ['source_document', 'similarity_score', 'matched_text', 'source_text', 'start_position', 'end_position']

class PlagiarismReportSerializer(serializers.ModelSerializer):
    matches = SimilarityMatchSerializer(many=True, read_only=True)
    document = DocumentSerializer(read_only=True)
    
    class Meta:
        model = PlagiarismReport
        fields = ['id', 'document', 'overall_similarity', 'total_matches', 'matches', 'created_at']