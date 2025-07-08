from django.urls import path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
from .views import UploadPaperView
from .views import AnalyzeExamView
from .views import RegisterView
from .views import MyTokenObtainPairView

urlpatterns = [
    path('login/', MyTokenObtainPairView.as_view(), name='token_obtain_pair'),

    path('refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('upload-paper/', UploadPaperView.as_view(), name='upload-paper'),
    path('analyze/', AnalyzeExamView.as_view(), name='analyze_exam'),
    path('register/', RegisterView.as_view(), name='register'),
    
]
