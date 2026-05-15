FROM python:3.11-slim

WORKDIR /app

# System deps for fitparse, lxml, psycopg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# withings-sync (a garminconnect dep) uses setuptools-scm and fails to build
# outside a git repo because it can't find the .VERSION file.
# SETUPTOOLS_SCM_PRETEND_VERSION tells setuptools-scm to use a hardcoded version
# instead of reading from git, which allows the build to succeed.
RUN grep -v "garminconnect" requirements.txt > /tmp/req_main.txt && \
    pip install --no-cache-dir -r /tmp/req_main.txt && \
    pip install --no-cache-dir "garth>=0.4.39" cloudscraper && \
    SETUPTOOLS_SCM_PRETEND_VERSION=2.7.5 pip install --no-cache-dir withings-sync && \
    pip install --no-cache-dir --no-deps garminconnect==0.2.20

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
