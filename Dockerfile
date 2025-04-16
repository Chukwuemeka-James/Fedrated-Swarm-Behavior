# Use the official TensorFlow image as the base image
FROM tensorflow/tensorflow:2.10.0-gpu

# Set environment variables to prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for pipenv
RUN apt-get update && apt-get install -y \
    python3-pip \
    curl \
    git \
    && pip install --upgrade pip \
    && pip install pipenv

# Copy Pipfile and Pipfile.lock first for better caching
COPY Pipfile Pipfile.lock ./

# Install Python dependencies using pipenv
RUN pipenv install --system --deploy

# Copy the rest of the project files
COPY . .

# Expose port for any external service (if needed)
EXPOSE 5000

# Set the default command to run the training
CMD ["python", "main.py"]
