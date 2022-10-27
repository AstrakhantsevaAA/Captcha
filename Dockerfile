FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04 AS base

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    cmake \
    cron \
    git \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libfreetype6-dev \
    libgl1-mesa-glx \
    libxrender1 \
    ninja-build \
    pkg-config \
    python3.9 \
    python3-pip \
    python3.9-dev \
    swig \
    wget \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

COPY requirements/torch.txt /requirements/
RUN pip3 install --no-cache-dir --upgrade pip==22.3 setuptools==59.5.0 wheel==0.37.0 \
    && pip3 install --no-cache-dir -r requirements/torch.txt --extra-index-url https://download.pytorch.org/whl/cu113

COPY requirements/base.txt /requirements/
RUN pip3 install --no-cache-dir -r requirements/base.txt
COPY requirements/ /requirements/
RUN pip3 install --no-cache-dir -r requirements/server.txt \
    && pip3 install --no-cache-dir -r requirements/train.txt

COPY models/release_v0.3.0 /models/release_v0.3.0

COPY captcha /captcha
COPY setup.py /setup.py
RUN pip3 install -e .

WORKDIR /captcha/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
