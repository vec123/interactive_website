# Build stage for frontend
FROM node:18 as frontend-builder

WORKDIR /app

COPY client/package*.json ./client/
RUN cd client && npm ci

COPY client ./client
RUN cd client && npm run build

# Backend stage
FROM python:3.9-slim

WORKDIR /app

# Install backend dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY server/ ./server

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/client/dist ./client/dist

EXPOSE 8000

CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
