FROM python:3.11-slim-bookworm

WORKDIR /app

COPY inference.py .
COPY model.py .
COPY runtime.py .
COPY code_maps ./code_maps
COPY requirements.txt .

# Install the CPU PyTorch build first; requirements.txt carries the same direct pin.
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Mount the direct weights root read-only at runtime.
ENV KIBATIN_WEIGHTS_DIR=/app/weights

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/healthz', timeout=4)"

# Keep one worker: every worker owns three complete multitask models.
CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "--threads=1", "--timeout=180", "inference:app"]
