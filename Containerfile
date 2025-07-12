# Use a stable NVIDIA CUDA runtime image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY ./requirements.txt /app/

# Upgrade pip and install Python libraries from requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the application code into the container
COPY ./app.py /app/

# Expose the port Gradio will run on
EXPOSE 7860

# The command to run when the container starts
CMD ["python", "app.py"]
