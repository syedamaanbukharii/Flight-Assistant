# Frontend image: Streamlit chat UI for the Flight & Travel Assistant.
# Build context is the repository root (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The frontend only needs Streamlit and an HTTP client to talk to the backend.
RUN pip install --no-cache-dir "streamlit>=1.36,<2.0" "httpx>=0.27,<1.0"

COPY frontend ./frontend

EXPOSE 8501

CMD ["streamlit", "run", "frontend/streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]
