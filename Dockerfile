# Container for Hugging Face Spaces (Docker SDK) or any cloud host. Runs the web app on port 7860.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=7860
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p app/uploads app/heatmaps && chmod -R a+rwX /app
EXPOSE 7860
CMD ["python", "run_app.py"]
