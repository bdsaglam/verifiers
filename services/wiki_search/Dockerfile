FROM python:3.12-slim

# Install Java and other dependencies
RUN apt-get update && apt-get install -y \
    openjdk-21-jdk \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Java options
ENV JAVA_OPTS="-Xms16g -Xmx32g -XX:MaxDirectMemorySize=16g"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 