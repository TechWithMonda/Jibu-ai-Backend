web: gunicorn jibu_backend.wsgi --bind 0.0.0.0:$PORT
worker: celery -A jibu_backend worker --loglevel=info
