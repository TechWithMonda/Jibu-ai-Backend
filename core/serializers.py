from rest_framework import serializers
from .models import UploadedPaper
from django.contrib.auth.models import User
# serializers.py
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth import authenticate

from .models import ExamPaper, SolutionView, UserActivity
from django.contrib.auth import get_user_model

User = get_user_model()

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
        # Map activity types to icon paths
        icons = {
            'upload': "M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12",
            'view': "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z",
            'feedback': "M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
        }
        return icons.get(obj.activity_type, "")
    
    def get_color(self, obj):
        # Map activity types to colors
        colors = {
            'upload': "blue",
            'view': "green",
            'feedback': "purple"
        }
        return colors.get(obj.activity_type, "gray")
class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    def validate(self, attrs):
        credentials = {
            'username': '',
            'password': attrs.get("password")
        }

        # This is asking for either username or email
        user_obj = User.objects.filter(email=attrs.get("username")).first() or \
                 User.objects.filter(username=attrs.get("username")).first()
        
        if user_obj:
            credentials['username'] = user_obj.username

        user = authenticate(**credentials)

        if not user:
            raise serializers.ValidationError('Invalid credentials')

        refresh = self.get_token(user)
        
        return {
            'refresh': str(refresh),
            'access': str(refresh.access_token),
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email
            }
        }


class UploadedPaperSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedPaper
        fields = ['id', 'file', 'uploaded_at']
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
            raise serializers.ValidationError("You must agree to the Terms of Service and Privacy Policy.")

        return data

    def create(self, validated_data):
        # Set email as username (Django default requires username)
        user = User.objects.create_user(
            username=validated_data['email'],
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data['full_name']
        )
        return user