# syntax=docker/dockerfile:1

# ---- Base ----
FROM python:3.11-slim

# Prevent Python from writing .pyc files & force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user
RUN useradd -m appuser
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better Docker layer caching
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir \
    streamlit==1.38.0 \
    requests>=2.31.0 \
    beautifulsoup4>=4.12.2 \
    cohere>=5.5.8 \
    openai>=1.51.0 \
    python-dotenv>=1.0.1 \
    pinecone>=3.0.0

# Copy app code
COPY app.py /app/app.py

# Make a writable data directory
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

# Expose Streamlit's default port
EXPOSE 8501

# Streamlit config: listen on all interfaces, keep CORS simple for local use
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]