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

from openai import OpenAI, OpenAIError 
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
from .models import ExamAnalysis 
import io
import logging
from datetime import datetime 

from .models import ExamPaper, SolutionView, UserActivity
from .serializers import ExamPaperSerializer, UserActivitySerializer
from django.db.models import Count, Avg, Sum



class DashboardAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        user = request.user
        
        # Calculate stats
        total_papers = ExamPaper.objects.filter(user=user).count()
        
        solutions_viewed = SolutionView.objects.filter(user=user).count()
        
        accuracy_rate = SolutionView.objects.filter(
            user=user,
            was_helpful__isnull=False
        ).aggregate(avg=Avg('was_helpful'))['avg'] or 0
        
        # Estimate time saved (assuming 30 minutes saved per solution viewed)
        time_saved = SolutionView.objects.filter(user=user).count() * 0.5
        
        # Get recent papers (last 5)
        recent_papers = ExamPaper.objects.filter(user=user).order_by('-uploaded_at')[:5]
        
        # Get recent activities (last 5)
        recent_activities = UserActivity.objects.filter(user=user).order_by('-created_at')[:5]
        
        # Serialize data
        paper_serializer = ExamPaperSerializer(recent_papers, many=True)
        activity_serializer = UserActivitySerializer(recent_activities, many=True)
        
        return Response({
            'stats': {
                'total_papers': total_papers,
                'solutions_viewed': solutions_viewed,
                'accuracy_rate': accuracy_rate,
                'time_saved': round(time_saved, 1)
            },
            'recentPapers': paper_serializer.data,
            'recentActivities': activity_serializer.data
        })# Add this if it exists

logger = logging.getLogger(__name__)

class AnalyzeExamView(APIView):
    permission_classes = [IsAuthenticated]
    
    # Allowed file types and sizes
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'application/pdf']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    # Model configurations
    MODEL_CONFIG = {
        'basic': {
            'prompt_template': "Provide concise answers to these exam questions:\n\n{text}",
            'model': "gpt-3.5-turbo",
            'max_tokens': 300
        },
        'standard': {
            'prompt_template': "Provide detailed step-by-step solutions to these exam questions:\n\n{text}",
            'model': "gpt-3.5-turbo",
            'max_tokens': 800
        },
        'advanced': {
            'prompt_template': """Provide comprehensive solutions with:
1. Multiple solution methods where applicable
2. Explanations of key concepts
3. References to relevant formulas/theorems
4. Practical applications

Questions:
{text}""",
            'model': "gpt-3.5-turbo",
            'max_tokens': 1500
        }
    }

    def post(self, request):
        try:
            # Validate input
            file = request.FILES.get('file')
            if not file:
                return Response({"error": "No file provided"}, status=400)

            # Validate file type and size
            self.validate_file(file)
            
            # Extract text
            start_time = datetime.now()
            text = self.extract_text_from_file(file)
            extraction_time = (datetime.now() - start_time).total_seconds()
            
            if not text.strip():
                return Response({"error": "Could not extract text from document"}, status=400)
            
            # Get model type (default to standard)
            model_type = request.data.get('model_type', 'standard').lower()
            if model_type not in self.MODEL_CONFIG:
                model_type = 'standard'
            
            # Call OpenAI API
            start_time = datetime.now()
            response = self.call_openai(text, model_type)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save to DB
            # ExamAnalysis.objects.create(
            #     user=request.user,
            #     original_filename=file.name,
            #     file_size=file.size,
            #     input_text=text[:5000],  # Store first 5000 chars
            #     model_type=model_type,
            #     response=response,
            #     extraction_time=extraction_time,
            #     processing_time=processing_time
            # )
            
            return Response({
                "result": response,
                "metadata": {
                    "model_used": model_type,
                    "extraction_time": extraction_time,
                    "processing_time": processing_time,
                    "total_time": extraction_time + processing_time
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing exam analysis: {str(e)}", exc_info=True)
            return Response({"error": "An error occurred during processing"}, status=500)

    def validate_file(self, file):
        """Validate file type and size"""
        if file.content_type not in self.ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported file type. Allowed types: {', '.join(self.ALLOWED_MIME_TYPES)}")
        
        if file.size > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum limit of {self.MAX_FILE_SIZE/1024/1024}MB")

    def extract_text_from_file(self, file):
        """Extract text from image or PDF using OCR"""
        try:
            file_content = file.read()
            
            if file.content_type == 'application/pdf':
                images = convert_from_bytes(file_content)
                text = ""
                for img in images:
                    text += pytesseract.image_to_string(img) + "\n"
                return text.strip()
            else:
                img = Image.open(io.BytesIO(file_content))
                return pytesseract.image_to_string(img)
                
        except Exception as e:
            logger.error(f"Error during text extraction: {str(e)}")
            raise ValueError("Could not extract text from document")

    def call_openai(self, text, model_type):
        """Call OpenAI API with the appropriate configuration"""
        try:
            config = self.MODEL_CONFIG[model_type]
            prompt = config['prompt_template'].format(text=text)

            client = OpenAI(api_key=settings.OPENAI_API_KEY)

            response = client.chat.completions.create(
                model=config['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful teaching assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config['max_tokens'],
                temperature=0.3,
                top_p=0.9
            )

            return response.choices[0].message.content

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise ValueError("Error communicating with AI service")

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