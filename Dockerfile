# Build frontend
FROM node:18 as frontend-builder
WORKDIR /app
COPY client/package*.json ./client/
RUN cd client && npm ci
COPY client ./client
RUN cd client && npm run build

# Backend
FROM python:3.9-slim
WORKDIR /app

# Install system libs for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, including utils
COPY server/ ./server

# Copy GP model files
COPY server/GP_models ./GP_models

# Copy .asf and .amc files
COPY server/01.asf server/01_01.amc ./

# Copy frontend build
COPY --from=frontend-builder /app/client/dist ./client/dist

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
