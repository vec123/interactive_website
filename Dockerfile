# Use official slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server

# Copy built frontend (make sure you ran `npm run build` first)
COPY client/dist ./client/dist

# Expose port FastAPI will run on
EXPOSE 8000

# Command to start FastAPI
CMD ["uvicorn", "server.server:app", "--host", "0.0.0.0", "--port", "8000"]
