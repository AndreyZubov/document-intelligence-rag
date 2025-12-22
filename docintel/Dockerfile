FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 docintel && chown -R docintel:docintel /app
USER docintel

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "docintel.api:app", "--host", "0.0.0.0", "--port", "8000"]
