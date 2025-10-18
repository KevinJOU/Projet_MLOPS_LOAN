FROM python:3.11-slim
WORKDIR /app
COPY requirements-infer.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY notebook/models/ notebook/models/
COPY app.py .
EXPOSE 8082
CMD ["python", "app.py"]
