FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer cache — only reinstalls if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY server/     ./server/
COPY openenv.yaml .
COPY inference.py .
COPY app.py .
COPY sre_triage.html .

# HuggingFace Spaces default port
EXPOSE 7860

# Healthcheck so HF Space knows when the container is ready
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')"

# Run FastAPI backend directly (simpler than Gradio for deployment)
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "7860"]
