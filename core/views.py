# At the top of views.py
import logging
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
from openai import OpenAI, OpenAIError
from PIL import Image
from django.contrib.auth import get_user_model
from pdf2image import convert_from_bytes
import pytesseract
import io


logger = logging.getLogger(__name__)

# Local imports at the bottom
from .models import (
    UploadedPaper, ExamAnalysis, ExamPaper, SolutionView, 
    UserActivity, Conversation, Message, Document, PlagiarismReport
)
from .serializers import (
    UploadedPaperSerializer, RegisterSerializer, MyTokenObtainPairSerializer,
    ExamPaperSerializer, UserActivitySerializer, ConversationSerializer,
    MessageSerializer, DocumentSerializer, PlagiarismReportSerializer,
    TutorRequestSerializer, TutorResponseSerializer
)
class DocumentViewSet(viewsets.ModelViewSet):
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request):
        """Create a new document"""
        title = request.data.get('title')
        content = request.data.get('content', '')
        file = request.FILES.get('file')
        
        if file:
            try:
                content = extract_text_from_file(file)
            except Exception as e:
                return Response(
                    {'error': f'Error processing file: {str(e)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
        
        document = Document.objects.create(
            title=title,
            content=content,
            file=file,
            uploaded_by=request.user if request.user.is_authenticated else None
        )
        
        serializer = self.get_serializer(document)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
    
    @action(detail=True, methods=['post'])
    def check_plagiarism(self, request, pk=None):
        """Check plagiarism for a specific document"""
        document = self.get_object()
        detector = PlagiarismDetector()
        
        try:
            report = detector.detect_plagiarism(document)
            serializer = PlagiarismReportSerializer(report)
            return Response(serializer.data)
        except Exception as e:
            return Response(
                {'error': f'Error during plagiarism check: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """Search documents"""
        query = request.query_params.get('q', '')
        if query:
            documents = Document.objects.filter(
                Q(title__icontains=query) | Q(content__icontains=query)
            )
        else:
            documents = Document.objects.all()
        
        serializer = self.get_serializer(documents, many=True)
        return Response(serializer.data)

class PlagiarismReportViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = PlagiarismReport.objects.all()
    serializer_class = PlagiarismReportSerializer
    
    def get_queryset(self):
        queryset = PlagiarismReport.objects.all()
        document_id = self.request.query_params.get('document_id')
        if document_id:
            queryset = queryset.filter(document__id=document_id)
        return queryset.order_by('-created_at')




class AITutorAPIView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        try:
            # Validate input
            request_serializer = TutorRequestSerializer(data=request.data)
            if not request_serializer.is_valid():
                return Response(
                    {'error': request_serializer.errors, 'status': 'error'},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            data = request_serializer.validated_data
            user = request.user
            
            # Get or create conversation
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
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                sender='user',
                content=data['message'],
                knowledge_level=data['knowledge_level']
            )
            
            # Generate AI response with error handling
            try:
                response_content = self.generate_response(
                    data['message'],
                    data['knowledge_level'],
                    data.get('action')
                )
            except Exception as e:
                logger.error(f"Response generation failed: {str(e)}")
                response_content = "I couldn't generate a response. Please try rephrasing your question."
            
            # Save bot response
            Message.objects.create(
                conversation=conversation,
                sender='bot',
                content=response_content,
                knowledge_level=data['knowledge_level']
            )
            
            # Prepare and validate response
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
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Create different prompts based on action and knowledge level
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
            text = self.extract_text_from_file(file)
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

    def extract_text_from_file(self, file):
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

    def call_openai(self, text, model_type):
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
        question = request.data.get("question")
        ai_response = f"AI-generated answer for: {question}"
        return Response({
            "question": question,
            "answer": ai_response
        })