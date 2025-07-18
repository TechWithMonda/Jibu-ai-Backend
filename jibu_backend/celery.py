# jibu_backend/celery.py
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'jibu_backend.settings')

app = Celery('jibu_backend')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
