# ONNX Runtime FastAPI serving image
# Usage:
#   docker build -t onnx-fastapi-serve .
#   docker run --rm -p 8000:8000 -e MODEL_PATH=/models/model.onnx -v $(pwd)/model.onnx:/models/model.onnx:ro onnx-fastapi-serve

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_PATH=/models/model.onnx \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

COPY serve_model.py /app/serve_model.py

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.30.6 \
        numpy==1.26.4 \
        onnxruntime==1.18.1 \
        pydantic==2.9.2 \
        python-multipart==0.0.9

# For GPU, replace the onnxruntime wheel with onnxruntime-gpu in a CUDA-matched base image.

RUN useradd -m appuser
USER appuser

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "serve_model:app", "--host", "0.0.0.0", "--port", "8000"]
