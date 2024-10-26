# Use an official Python runtime as a parent image
FROM python:3.11-slim

WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY src /app/src

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variable for unbuffered Python output
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["streamlit", "run", "src/app.py"]
