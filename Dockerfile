FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformers model at build time so the container
# starts instantly. jobbert (~440 MB) is excluded — ATSScorer falls back
# to static keyword extraction when it can't be loaded.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Prevent runtime attempts to fetch uncached models from the Hub.
# all-MiniLM-L6-v2 is cached above and loads fine; jobbert is not cached
# and fails gracefully inside ATSScorer._load_models().
ENV HF_HUB_OFFLINE=1
ENV PYTHONUNBUFFERED=1

COPY . .

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
