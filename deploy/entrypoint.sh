#!/usr/bin/env bash
set -e

echo "Starting TDC-Studio Serving Service..."
exec uvicorn tdc_studio.serving.app:app --host 0.0.0.0 --port 8000 --workers 2
