# CUDA 지원을 위한 NVIDIA 베이스 이미지 선택 (Ubuntu 20.04 기반)
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libglib2.0-0 \
    tzdata \
    && ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone \    
    && rm -rf /var/lib/apt/lists/*

# Python 3를 기본 python으로 설정
RUN ln -s /usr/bin/python3 /usr/bin/python

# pip 업그레이드
RUN pip install --upgrade pip

# PyTorch 및 관련 라이브러리 설치 (CUDA 11.6)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

# CuPy CUDA 바이너리 설치
RUN pip install cupy-cuda116

# 추가 Python 패키지 설치
RUN pip install numpy pandas matplotlib scipy opencv-python tensorboardX

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본 명령 설정
CMD ["bash"]
