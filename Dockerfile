FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone darknet and build with shared lib + OpenCV + Python bindings
RUN git clone https://github.com/AlexeyAB/darknet.git && \
    cd darknet && \
    sed -i 's/GPU=0/GPU=0/' Makefile && \
    sed -i 's/CUDNN=0/CUDNN=0/' Makefile && \
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile && \
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile && \
    make -j$(nproc)

# Add darknet to Python path
ENV PYTHONPATH=/app/darknet:$PYTHONPATH

# Copy your app code
COPY . .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Download yolov4 weights
RUN mkdir -p cfg && \
    curl -L https://pjreddie.com/media/files/yolov4.weights -o yolov4.weights

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
