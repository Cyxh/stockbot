FROM python:3.11-slim

WORKDIR /app

# Timezone data for market hours detection
RUN apt-get update && apt-get install -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create cache directory
RUN mkdir -p .data_cache

CMD ["python", "-u", "main.py", "live"]
