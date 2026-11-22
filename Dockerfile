# Container for the Gradio face-generation app (CPU only).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PORT=7860

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gan_faces ./gan_faces
COPY models ./models
COPY app.py .

EXPOSE 7860
CMD ["python", "app.py"]
