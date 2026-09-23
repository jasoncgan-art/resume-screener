# CPU-only serving image for resume-screener.
#
# Artifacts (artifacts/biencoder.pt, artifacts/dataset.pt) are intentionally
# NOT baked into the image: they are produced by `python train.py` and should
# be bind-mounted at run time so the image stays small and model versions
# stay decoupled from the serving code.
#
# Build:  docker build -f serving/Dockerfile -t resume-screener .
# Run:    docker run -p 8000:8000 -v $PWD/artifacts:/app/artifacts resume-screener

FROM python:3.12-slim

WORKDIR /app

# torch CPU wheel index keeps the install small (~200 MB, no CUDA bundle).
COPY requirements.txt serving/requirements.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple/ \
        -r requirements.txt -r serving/requirements.txt

COPY dataset.py models.py screen.py ./
COPY serving/ ./serving/

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
