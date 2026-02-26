FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create necessary directories
RUN mkdir -p models data/processed data/raw reports results /var/log

# Change ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Set environment variable for local API connection
ENV API_BASE_URL=http://localhost:8000

# Create startup script
RUN echo '#!/bin/bash\n\
# Start API in background\n\
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 &\n\
# Wait for API to start\n\
sleep 10\n\
# Start Streamlit on Render port\n\
streamlit run streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0' > /app/start.sh && \
chmod +x /app/start.sh

# Use the startup script
CMD ["/app/start.sh"]
