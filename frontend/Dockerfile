# Use a lightweight version of Python
FROM python:3.10-slim

# Set the working directory for the UI
WORKDIR /app

# Copy requirements from the root (since context is .)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project structure
# This ensures app.py can find other modules if needed
COPY . .

# Streamlit uses port 8501 by default
EXPOSE 8501

# Command to launch the Streamlit dashboard
# We point to the specific path where app.py lives
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]