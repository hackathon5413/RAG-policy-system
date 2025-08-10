
FROM python:3.13-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    libpoppler-cpp-dev \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .


RUN mkdir -p /app/data

EXPOSE 8080


CMD ["python", "server.py"]
