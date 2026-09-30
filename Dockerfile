# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8502

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
