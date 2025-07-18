from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import MyTokenObtainPairView
from . import views
# Local imports
from .views import (
    UploadPaperView,
    AnalyzeExamView,
    RegisterView,
    DashboardAPIView,
    AITutorAPIView,
    DocumentViewSet,
   
    GenerateQuizQuestions,
    VoiceQueryView,
    paystack_webhook,
    VerifyPaymentView,
    TaskStatusView
    
)





urlpatterns = [
    path('login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),  # ✅ Use your custom view
    path('dashboard/', DashboardAPIView.as_view(), name='dashboard'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('upload-paper/', UploadPaperView.as_view(), name='upload-paper'),
    path('analyze/', AnalyzeExamView.as_view(), name='analyze_exam'),
    path('register/', RegisterView.as_view(), name='register'),
    path('tutor/', AITutorAPIView.as_view(), name='ai_tutor_api'),
    path('generate-quiz/', GenerateQuizQuestions.as_view(), name='generate_quiz'), 
    path("voice-query/", VoiceQueryView.as_view(), name="voice-query"),
    path('verify-payment/', VerifyPaymentView.as_view(), name='verify_payment'),
     path('webhook/paystack/', paystack_webhook),
     path('task-status/<str:task_id>/', TaskStatusView.as_view()),
]