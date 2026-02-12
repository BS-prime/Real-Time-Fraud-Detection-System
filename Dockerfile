FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# XGBoost runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install project dependencies first for better layer caching
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# Copy runtime assets used by the API
COPY artifacts ./artifacts

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
