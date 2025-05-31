# Use official python base image
FROM python:3.9-slim

# Install system dependencies for darknet and python opencv
RUN apt-get update && apt-get install -y \
    git build-essential cmake libopencv-dev pkg-config libjpeg-dev libpng-dev libtiff-dev libavformat-dev libavcodec-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev python3-opencv wget curl \
    && rm -rf /var/lib/apt/lists/*

# Clone and build darknet for Python wrapper
RUN git clone https://github.com/AlexeyAB/darknet.git /darknet

WORKDIR /darknet
RUN sed -i 's/GPU=0/GPU=0/' Makefile && \
    sed -i 's/CUDNN=0/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile

RUN make

# Copy your config files and weights into /app/cfg
WORKDIR /app
COPY cfg ./cfg
COPY app.py .
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download yolov2 weights to cfg/
RUN mkdir -p cfg && \
    curl -L https://pjreddie.com/media/files/yolov2.weights -o cfg/yolov2.weights

# Set environment variables for darknet python wrapper
ENV DARKNET_PATH=/darknet

# Add darknet python wrapper to python path
ENV PYTHONPATH=/darknet:$PYTHONPATH

# Expose port
EXPOSE 8000

# Command to run FastAPI app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
