#!/bin/sh
echo "✅ PORT is: ${PORT}"

# Stop if PORT is empty
if [ -z "$PORT" ]; then
  echo "❌ ERROR: PORT is not set!"
  exit 1
fi

# Apply migrations
python manage.py migrate --noinput

# Collect static
python manage.py collectstatic --noinput --clear

# Start Gunicorn
exec gunicorn jibu_backend.wsgi:application --bind 0.0.0.0:$PORT
