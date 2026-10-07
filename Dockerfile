FROM python:3.11-slim

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: requirements.txt contains a custom index url for CUDA 12.6
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command to run the test suite
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
