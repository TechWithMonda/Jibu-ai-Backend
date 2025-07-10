from django.urls import path,include
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import UploadPaperView
from .views import AnalyzeExamView
from .views import RegisterView
from .views import MyTokenObtainPairView
from .views import DashboardAPIView
from .views import AITutorAPIView
from rest_framework.routers import DefaultRouter
from .import  views


router = DefaultRouter()
router.register(r'documents', views.DocumentViewSet)
router.register(r'reports', views.PlagiarismReportViewSet)





urlpatterns = [
    path('login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('dashboard/', DashboardAPIView.as_view(), name='dashboard'),
    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('upload-paper/', UploadPaperView.as_view(), name='upload-paper'),
    path('analyze/', AnalyzeExamView.as_view(), name='analyze_exam'),
    path('register/', RegisterView.as_view(), name='register'),
    path('tutor/', AITutorAPIView.as_view(), name='ai_tutor_api'),
    path('plagiarism-check/', views.UploadAndCheckPlagiarism.as_view(), name="check"),
    path('', include(router.urls)),
]