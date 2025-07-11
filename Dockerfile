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

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code, including utils and GP_models
COPY server/ ./server

# Copy frontend build output
COPY --from=frontend-builder /app/client/dist ./client/dist

# Expose port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
