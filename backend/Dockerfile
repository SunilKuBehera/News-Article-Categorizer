# Use a slim Python image for a smaller, faster container
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements from the root (since context is .)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data ---
# This ensures the container has the 'punkt_tab' resource ready to go
RUN python -m nltk.downloader punkt_tab stopwords

# Copy the entire project structure to maintain relative imports
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI server
# We use 'backend.main:app' because 'main.py' is inside the 'backend' folder
# relative to the /app (root) directory.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]