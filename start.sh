#!/bin/bash
# Startup script for Railway deployment

echo "🚀 Starting Bibliometric Analysis CLI API"
echo "📍 PORT variable: ${PORT}"
echo "🔧 Python version: $(python --version)"

# Check if PORT is set, default to 8000
if [ -z "$PORT" ]; then
    echo "⚠️  PORT not set, defaulting to 8000"
    export PORT=8000
fi

echo "✅ Starting gunicorn on port ${PORT}"

# Start gunicorn with the PORT variable
exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 2 \
    --timeout 300 \
    --log-level debug \
    --access-logfile - \
    --error-logfile - \
    --capture-output \
    api_server:app
