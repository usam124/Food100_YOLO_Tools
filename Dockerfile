FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update --fix-missing && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        libopencv-dev \
        python3-dev \
        python3-pip \
        python3-opencv \
        pkg-config \
        libgtk-3-dev \
        software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone darknet repo
RUN git clone https://github.com/AlexeyAB/darknet.git

WORKDIR /app/darknet

# Enable OpenCV and build shared library for Python bindings
RUN sed -i 's/GPU=0/GPU=0/' Makefile && \
    sed -i 's/CUDNN=0/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile && \
    make -j$(nproc) && \
    cp libdarknet.so /app/ && \
    cp darknet.py /app/ && \
    cp -r cfg /app/ && \
    cp -r data /app/

# Add darknet to Python path and ensure libdarknet.so is found
ENV PYTHONPATH=/app:$PYTHONPATH
ENV LD_LIBRARY_PATH=/app:$LD_LIBRARY_PATH

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download YOLOv4 weights
RUN curl -L https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -o yolov4.weights

# Ensure permissions for libdarknet.so
RUN chmod +x /app/libdarknet.so

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
