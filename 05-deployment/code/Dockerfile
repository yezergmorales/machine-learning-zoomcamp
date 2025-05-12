# Use an official Python runtime as a parent image
ARG PYTHON_VERSION=3.12.3
FROM python:${PYTHON_VERSION}-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and model file into the container at /app
COPY predict.py .
COPY model_C=1.0.bin .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable (optional, good practice)
ENV PYTHONUNBUFFERED=1

# Run uvicorn when the container launches
# Use 0.0.0.0 to ensure the server is accessible from outside the container
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]