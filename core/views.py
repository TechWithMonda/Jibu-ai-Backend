from django.shortcuts import render

# Create your views here.
from rest_framework import generics, permissions
from .models import UploadedPaper
from .serializers import UploadedPaperSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated

from rest_framework import generics
from .serializers import RegisterSerializer
from django.contrib.auth.models import User


# views.py
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import MyTokenObtainPairSerializer
from rest_framework import status
from django.conf import settings
import openai
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import io
from .models import ExamAnalysis  # Add this if it exists


class AnalyzeExamView(APIView):
    def post(self, request):
        # Extract text from file (image/PDF)
        file = request.FILES.get('file')
        if not file:
            return Response({"error": "No file provided"}, status=400)

        # OCR: Extract text using Tesseract
        text = self.extract_text_from_file(file)
        
        # Call OpenAI API
        model_type = request.data.get('model_type', 'standard')
        response = self.call_openai(text, model_type)
        
        # Save to DB (optional)
        ExamAnalysis.objects.create(
            user=request.user,
            input_text=text,
            model_type=model_type,
            response=response
        )
        
        return Response({"result": response})

    def extract_text_from_file(self, file):
        if file.name.lower().endswith('.pdf'):
            images = convert_from_bytes(file.read())
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img)
            return text
        else:
            img = Image.open(io.BytesIO(file.read()))
            return pytesseract.image_to_string(img)

    def call_openai(self, text, model_type):
        openai.api_key = settings.OPENAI_API_KEY
        
        # Customize prompt based on model type
        if model_type == "basic":
            prompt = f"Answer concisely: {text}"
            model = "gpt-3.5-turbo"
        elif model_type == "advanced":
            prompt = f"Explain in detail with references: {text}"
            model = "gpt-4-turbo"
        else:
            prompt = f"Provide step-by-step solution: {text}"
            model = "gpt-4"
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3  # Less creative, more factual
        )
        return response.choices[0].message.content
class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer

class UploadPaperView(generics.CreateAPIView):
    serializer_class = UploadedPaperSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
class AnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Example request body: {"question": "Explain photosynthesis"}
        question = request.data.get("question")
        
        # TODO: Integrate OpenAI or custom AI logic here
        ai_response = f"AI-generated answer for: {question}"

        return Response({
            "question": question,
            "answer": ai_response
        })