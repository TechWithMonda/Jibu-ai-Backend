#!/bin/sh
echo "✅ PORT value is: $PORT"

# Apply database migrations
python manage.py migrate --noinput

# Collect static files
python manage.py collectstatic --noinput

# Run Gunicorn with proper port expansion
exec gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:${PORT:-8000}
