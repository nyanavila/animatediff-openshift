# Use a newer NVIDIA CUDA runtime image to support torch>=2.4.0
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

# Install system dependencies, including git
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git build-essential cmake curl unzip ffmpeg \
    libsm6 libxext6 libgl1-mesa-glx ninja-build \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set the working directory
WORKDIR /app

# Clone the official repository to get the 'wan' library using the git protocol
RUN git clone git://github.com/ali-vilab/VGen.git

# Copy our custom requirements file
COPY ./requirements.txt /app/

# Install Python libraries and immediately clean up the cache to save space
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy our custom application script
COPY ./app.py /app/

# Expose the port Gradio will run on
EXPOSE 7860

# The command to run when the container starts
CMD ["python", "app.py"]
