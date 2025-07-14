from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import MyTokenObtainPairView
# Local imports
from .views import (
    UploadPaperView,
    AnalyzeExamView,
    RegisterView,
    DashboardAPIView,
    AITutorAPIView,
    UploadAndCheckPlagiarism,
    DocumentViewSet,
    PlagiarismReportViewSet,
    GenerateQuizQuestions,
    VoiceQueryView,

)

router = DefaultRouter()
router.register(r'documents', DocumentViewSet, basename='documents')
router.register(r'reports', PlagiarismReportViewSet, basename='reports')

urlpatterns = [
    path('login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),  # ✅ Use your custom view
    path('dashboard/', DashboardAPIView.as_view(), name='dashboard'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('upload-paper/', UploadPaperView.as_view(), name='upload-paper'),
    path('analyze/', AnalyzeExamView.as_view(), name='analyze_exam'),
    path('register/', RegisterView.as_view(), name='register'),
    path('tutor/', AITutorAPIView.as_view(), name='ai_tutor_api'),
    path('plagiarism-check/', UploadAndCheckPlagiarism.as_view(), name="check"),
    path('generate-quiz/', GenerateQuizQuestions.as_view(), name='generate_quiz'), 
    path("voice-query/", VoiceQueryView.as_view(), name="voice-query"),
    path('', include(router.urls)),
]