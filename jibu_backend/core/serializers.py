from rest_framework import serializers
from .models import UploadedPaper
from django.contrib.auth.models import User
# serializers.py
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from django.contrib.auth import authenticate


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