release: python manage.py migrate

web: gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:$PORT --timeout 120 --threads 2 --workers 2
