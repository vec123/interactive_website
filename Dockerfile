# Use a minimal Python image
FROM python:3.9-slim

WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server

# Copy frontend build
COPY client/dist ./client/dist

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
