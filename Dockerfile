# Fusion Risk — single container running FastAPI (port 8000) + Streamlit (port 7860).
# 7860 is the port Hugging Face Spaces expects; Streamlit is the public UI.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1 is required by xgboost / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# HF Spaces sets $PORT; default to 7860 for the Streamlit UI
ENV PORT=7860
EXPOSE 7860 8000

RUN chmod +x start.sh
CMD ["./start.sh"]
