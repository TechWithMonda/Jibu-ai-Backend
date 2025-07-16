# views.py
import logging
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from datetime import datetime
from django.conf import settings
from django.db.models import Avg, Q
from rest_framework import generics, permissions, status, viewsets
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import action
from rest_framework_simplejwt.views import TokenObtainPairView
import openai
from openai import OpenAIError
import mimetypes
import os
import tempfile
import pyttsx3
from io import BytesIO
import numpy as np
from PIL import Image
from django.contrib.auth import get_user_model
from pdf2image import convert_from_bytes
import pytesseract
import io
from pydub import AudioSegment
from django.db import transaction  # Add this import
import re  # Add this import

# Initialize OpenAI client once
openai.api_key = settings.OPENAI_API_KEY
logger = logging.getLogger(__name__)

# Get User model
User = get_user_model()

# Local imports
from .models import (
    UploadedPaper, ExamAnalysis, ExamPaper, SolutionView,
    UserActivity, Conversation, Message, Document, UploadedDocument
)
from .serializers import (
    UploadedPaperSerializer, RegisterSerializer, MyTokenObtainPairSerializer,
    ExamPaperSerializer, UserActivitySerializer, ConversationSerializer,
    MessageSerializer, DocumentSerializer, 
    TutorRequestSerializer, TutorResponseSerializer
)
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
import json
from django.utils import timezone
from io import BytesIO
import requests
from rest_framework.parsers import JSONParser
import base64
from django.utils.timezone import now, timedelta
from .utils import validate_audio_file, cleanup_temp_files
from django.db.models import Count
from .models import Payment, UserProfile
from pydub.utils import which
from django.views.decorators.csrf import csrf_exempt
from django.db.models.functions import TruncDate
import json
import hashlib
import hmac
from .models import PremiumUser  # Create this model to track who paid
from django.http import JsonResponse
from rest_framework.authentication import TokenAuthentication
import logging
from django.contrib.auth.hashers import make_password
logger = logging.getLogger(__name__)
from rest_framework_simplejwt.authentication import JWTAuthentication


PAYSTACK_SECRET_KEY = os.getenv('PAYSTACK_SECRET_KEY')
BASE_URL = "https://api.paystack.co"

class VerifyPaymentView(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            reference = request.data.get('reference')
            email = request.data.get('email')
            
            if not reference or not email:
                return Response({"error": "Missing reference or email"}, status=400)

            # Get or create user
            user, created = User.objects.get_or_create(
                email=email,
                defaults={
                    'username': email,
                    'password': make_password(User.objects.make_random_password())
                }
            )

            # Call Paystack API
            headers = {"Authorization": f"Bearer {PAYSTACK_SECRET_KEY}"}
            response = requests.get(f"{BASE_URL}/transaction/verify/{reference}", headers=headers, timeout=10)
            data = response.json()

            if not response.ok:
                return Response({"error": data.get('message', 'Paystack verification failed')}, status=400)

            if data.get('status') is not True or data['data'].get('status') != 'success':
                return Response({"error": "Payment not successful"}, status=400)

            payment_data = data['data']

            # Save Payment record
            Payment.objects.update_or_create(
                reference=reference,
                defaults={
                    'user': user,
                    'amount': payment_data['amount'] / 100,
                    'currency': payment_data.get('currency', 'NGN'),
                    'status': 'success',
                    'paid_at': timezone.now(),
                    'gateway_response': payment_data
                }
            )

            # Save PremiumUser
            PremiumUser.objects.update_or_create(
                user=user,
                defaults={
                    'email': email,
                    'plan': 'Premium',
                    'reference': reference,
                    'activated_at': timezone.now()
                }
            )

            # Update UserProfile
            UserProfile.objects.update_or_create(
                user=user,
                defaults={'is_paid': True}
            )

            return Response({
                "status": "success",
                "user_id": user.id,
                "email": user.email
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"Paystack API error: {str(e)}")
            return Response({"error": "Payment verification service unavailable"}, status=503)
        except Exception as e:
            logger.exception("Payment verification failed")
            return Response({"error": "Internal server error"}, status=500)
@csrf_exempt
def paystack_webhook(request):
    if request.method != 'POST':
        return JsonResponse({"status": "method not allowed"}, status=405)

    try:
        payload = request.body
        signature = request.headers.get('x-paystack-signature')
        
        if not signature:
            logger.warning("Missing Paystack signature in webhook")
            return JsonResponse({"status": "forbidden"}, status=403)

        # Verify signature
        expected_signature = hmac.new(
            settings.PAYSTACK_SECRET_KEY.encode(),
            payload,
            hashlib.sha512
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_signature):
            logger.warning("Invalid Paystack webhook signature")
            return JsonResponse({"status": "forbidden"}, status=403)

        data = json.loads(payload)
        event = data.get('event')
        
        logger.info(f"Paystack webhook received: {event}")

        if event == 'charge.success':
            payment_data = data['data']
            email = payment_data['customer']['email']
            reference = payment_data['reference']
            amount = payment_data['amount'] / 100

            # Get or create user
            user, _ = User.objects.get_or_create(
                email=email,
                defaults={'username': email}
            )

            # Create payment record
            Payment.objects.update_or_create(
                reference=reference,
                defaults={
                    'user': user,
                    'amount': amount,
                    'currency': payment_data.get('currency', 'NGN'),
                    'status': 'success',
                    'paid_at': timezone.now(),
                    'gateway_response': payment_data
                }
            )

            # Update premium status - using get_or_create to handle missing user_id
            premium_user, created = PremiumUser.objects.get_or_create(
                email=email,
                defaults={
                    'user': user,
                    'plan': 'Premium',
                    'reference': reference,
                    'activated_at': timezone.now()
                }
            )
            
            if not created:
                premium_user.user = user
                premium_user.save()

            return JsonResponse({"status": "success"})

    except Exception as e:
        logger.exception("Webhook processing error")
        return JsonResponse({"status": "error"}, status=500)

ffmpeg_path = which("ffmpeg")
ffprobe_path = which("ffprobe")

AudioSegment.converter = ffmpeg_path
AudioSegment.ffprobe   = ffprobe_path
class VoiceQueryView(APIView):
    def post(self, request, *args, **kwargs):
        try:
            # Validate input
            audio_file = request.FILES.get('audio')
            language = request.data.get('language', 'en').lower()
            
            if not audio_file:
                return Response({'error': 'No audio file provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            if language not in ['en', 'sw']:
                return Response({'error': 'Unsupported language'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate audio file
            validation_error = validate_audio_file(audio_file)
            if validation_error:
                return Response({'error': validation_error}, status=status.HTTP_400_BAD_REQUEST)

            # Transcribe audio using Whisper
            try:
                (...)
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

                input_audio = BytesIO(audio_file.read())
                input_audio.name = audio_file.name
                input_audio.seek(0)
                                
                audio = AudioSegment.from_file(input_audio)
                output_buffer = BytesIO()
                audio.export(output_buffer, format="mp3")
                output_buffer.name = "converted.mp3"
                output_buffer.seek(0)

                openai.api_key = settings.OPENAI_API_KEY
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=output_buffer,
                    language=language
                )

                user_text = transcript.text
            except Exception as e:
                return Response({'error': f'Transcription failed: {str(e)}'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Generate AI response
            try:
                system_prompt = (
                    "You are a helpful Swahili tutor." if language == 'sw' 
                    else "You are a helpful English tutor."
                )
                response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.7,
                max_tokens=500
            )
                answer = response.choices[0].message.content.strip()
            except Exception as e:
                return Response({'error': f'AI response failed: {str(e)}'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Generate TTS response
            try:
                engine = pyttsx3.init()
                
                # Configure voice based on language
                voices = engine.getProperty('voices')
                if language == 'sw':
                    # Try to find a Swahili-compatible voice
                    swahili_voices = [v for v in voices if 'swahili' in v.name.lower() or 'africa' in v.name.lower()]
                    if swahili_voices:
                        engine.setProperty('voice', swahili_voices[0].id)
                    else:
                        engine.setProperty('voice', voices[1].id)  # Fallback
                else:
                    english_voices = [v for v in voices if 'english' in v.name.lower()]
                    if english_voices:
                        engine.setProperty('voice', english_voices[0].id)
                
                # Adjust speech parameters
                engine.setProperty('rate', 150)  # Slower speech for educational content
                engine.setProperty('volume', 0.9)  # Slightly lower volume
                
                # Generate audio file
        
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
                    wav_path = wav_file.name

                    engine.save_to_file(answer, wav_path)
                    engine.runAndWait()

                    # Convert WAV to MP3
                    mp3_buffer = BytesIO()
                    sound = AudioSegment.from_wav(wav_path)
                    sound.export(mp3_buffer, format="mp3")
                    mp3_buffer.seek(0)

                    audio_base64 = base64.b64encode(mp3_buffer.read()).decode('utf-8')

                    # Clean up temp files
                    cleanup_temp_files(wav_path)
                    
            except Exception as e:
                return Response({'error': f'Audio generation failed: {str(e)}'}, 
                              status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response({
                "user_question": user_text,
                "text_response": answer,
                "audio_response": audio_base64,
                "language": language
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({'error': f'Unexpected error: {str(e)}'}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
class GenerateQuizQuestions(APIView):
    def post(self, request, *args, **kwargs):
        user = request.user
        try:
            topic = request.data.get('topic')
            difficulty = request.data.get('difficulty', 'beginner')
            num_questions = request.data.get('num_questions', 5)

            if not topic:
                return Response(
                    {"error": "Topic is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            prompt = f"""Generate {num_questions} multiple choice questions about {topic} at {difficulty} level.
Format each question exactly like this:

Q: What is the capital of France?
A: Paris
B: London
C: Berlin
D: Madrid
Correct: A

Include exactly 4 options (A-D) and mark the correct one.
Return only the questions in this format."""
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000
            )

            content = response.choices[0].message.content
            questions = self.parse_questions(content, topic)
            return Response({'questions': questions})

        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def parse_questions(self, text, topic):
        questions = []
        current_q = {}
        lines = text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if line.startswith('Q:'):
                if current_q:
                    questions.append(current_q)
                current_q = {
                    'question': line[3:].strip(),
                    'options': [],
                    'topic': topic
                }
            elif line.startswith(('A:', 'B:', 'C:', 'D:')):
                current_q['options'].append(line[3:].strip())
            elif line.startswith('Correct:'):
                correct_letter = line[8:].strip()
                letter_index = ord(correct_letter.upper()) - ord('A')
                current_q['correctAnswer'] = letter_index

        if current_q:
            questions.append(current_q)
        return questions
def extract_text_from_file(file):
    try:
        file_content = file.read()
        if file.content_type == 'application/pdf':
            images = convert_from_bytes(file_content)
            return "\n".join(pytesseract.image_to_string(img) for img in images).strip()
        img = Image.open(io.BytesIO(file_content))
        return pytesseract.image_to_string(img)
    except Exception as e:
        logger.error(f"Error during text extraction: {str(e)}")
        raise ValueError("Could not extract text from document")

class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        """Only show documents belonging to the current user"""
        return self.queryset.filter(uploaded_by=self.request.user)

    def perform_create(self, serializer):
        """Automatically set the uploaded_by field to current user"""
        serializer.save(uploaded_by=self.request.user)

    @action(detail=True, methods=['post'])

    @action(detail=False, methods=['get'])
    def search(self, request):
        query = request.query_params.get('q', '')
        if query:
            documents = Document.objects.filter(
                Q(title__icontains=query) | Q(content__icontains=query)
            )
        else:
            documents = Document.objects.all()
        
        serializer = self.get_serializer(documents, many=True)
        return Response(serializer.data)



class AITutorAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            request_serializer = TutorRequestSerializer(data=request.data)
            if not request_serializer.is_valid():
                return Response(
                    {'error': request_serializer.errors, 'status': 'error'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            data = request_serializer.validated_data
            user = request.user
            
            if data.get('conversation_id'):
                conversation = Conversation.objects.get(
                    id=data['conversation_id'],
                    user=user
                )
            else:
                conversation = Conversation.objects.create(
                    user=user,
                    title=data['message'][:50] + '...'
                )
            
            Message.objects.create(
                conversation=conversation,
                sender='user',
                content=data['message'],
                knowledge_level=data['knowledge_level']
            )
            
            try:
                response_content = self.generate_response(
                    data['message'],
                    data['knowledge_level'],
                    data.get('action')
                )
            except Exception as e:
                logger.error(f"Response generation failed: {str(e)}")
                response_content = "I couldn't generate a response. Please try rephrasing your question."
            
            Message.objects.create(
                conversation=conversation,
                sender='bot',
                content=response_content,
                knowledge_level=data['knowledge_level']
            )
            
            response_data = {
                'response': response_content,
                'conversation_id': conversation.id,
                'status': 'success'
            }
            
            response_serializer = TutorResponseSerializer(data=response_data)
            if not response_serializer.is_valid():
                raise Exception("Invalid response format")
                
            return Response(response_serializer.validated_data)
            
        except Exception as e:
            logger.error(f"Error in AITutorAPIView: {str(e)}", exc_info=True)
            return Response(
                {'error': str(e), 'status': 'error'},
                status=status.HTTP_400_BAD_REQUEST
            )
    
    def generate_response(self, message, knowledge_level, action=None):
        try:
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            
            if action == 'related':
                prompt = f"Provide 3-5 topics closely related to: {message}"
            elif action == 'simplify':
                prompt = f"Explain this in simple terms suitable for a beginner: {message}"
            elif action == 'example':
                prompt = f"Provide a clear example illustrating: {message}"
            elif action == 'practice':
                prompt = f"Generate a practice question about: {message} (include answer)"
            else:
                level_prompts = {
                    'beginner': "Explain this in basic terms: {message}",
                    'intermediate': "Provide a detailed explanation of: {message}",
                    'advanced': "Give an advanced technical analysis of: {message}"
                }
                prompt = level_prompts.get(knowledge_level, level_prompts['intermediate']).format(message=message)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable tutor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            return "I encountered an error generating a response. Please try again."

class DashboardAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        total_papers = ExamPaper.objects.filter(user=user).count()
        solutions_viewed = SolutionView.objects.filter(user=user).count()
        accuracy_rate = SolutionView.objects.filter(user=user, was_helpful__isnull=False).aggregate(avg=Avg('was_helpful'))['avg'] or 0
        time_saved = solutions_viewed * 0.5

        recent_papers = ExamPaper.objects.filter(user=user).order_by('-uploaded_at')[:5]
        recent_activities = UserActivity.objects.filter(user=user).order_by('-created_at')[:5]

        return Response({
            'stats': {
                'total_papers': total_papers,
                'solutions_viewed': solutions_viewed,
                'accuracy_rate': accuracy_rate,
                'time_saved': round(time_saved, 1)
            },
            'recentPapers': ExamPaperSerializer(recent_papers, many=True).data,
            'recentActivities': UserActivitySerializer(recent_activities, many=True).data
        })

class AnalyzeExamView(APIView):
    permission_classes = [IsAuthenticated]
    ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'application/pdf']
    MAX_FILE_SIZE = 10 * 1024 * 1024

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
            file = request.FILES.get('file')
            if not file:
                return Response({"error": "No file provided"}, status=400)
            self.validate_file(file)

            start_time = datetime.now()
            text = extract_text_from_file(file)
            extraction_time = (datetime.now() - start_time).total_seconds()

            if not text.strip():
                return Response({"error": "Could not extract text from document"}, status=400)

            model_type = request.data.get('model_type', 'standard').lower()
            if model_type not in self.MODEL_CONFIG:
                model_type = 'standard'

            start_time = datetime.now()
            response = self.call_openai(text, model_type)
            processing_time = (datetime.now() - start_time).total_seconds()

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
        if file.content_type not in self.ALLOWED_MIME_TYPES:
            raise ValueError(f"Unsupported file type. Allowed types: {', '.join(self.ALLOWED_MIME_TYPES)}")
        if file.size > self.MAX_FILE_SIZE:
            raise ValueError("File size exceeds maximum limit")

    def call_openai(self, text, model_type):
        try:
            config = self.MODEL_CONFIG[model_type]
            prompt = config['prompt_template'].format(text=text)
            client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
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
    permission_classes = [permissions.AllowAny] 

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = RegisterSerializer
    permission_classes = [permissions.AllowAny]

class UploadPaperView(generics.CreateAPIView):
    serializer_class = UploadedPaperSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class AnalyzeView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        question = request.data.get("question")
        ai_response = f"AI-generated answer for: {question}"
        return Response({
            "question": question,
            "answer": ai_response
        })