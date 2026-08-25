FROM python:3.11-slim

# libsndfile нужен для soundfile/librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway передают порт через переменную окружения PORT
ENV PORT=8000
EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
