FROM nvidia/cuda:9.2-cudnn7-devel

RUN apt-get update --allow-insecure-repositories \
    && apt-get install -y --no-install-recommends  \
    python3-dev \
    git \
    wget \
    curl \
    cmake \
    libblas3 \
    libblas-dev \
    && curl -kL https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install --upgrade pip setuptools

# OpenCV
RUN apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

# MKL
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB \
    && wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list \
    && apt update \
    && apt install -y intel-mkl-2019.4-070 \
    && rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

ENV MKL_ROOT_DIR=/opt/intel/mkl
ENV LD_LIBRARY_PATH=$MKL_ROOT_DIR/lib/intel64:/opt/intel/lib/intel64_lin:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=$MKL_ROOT_DIR/lib/intel64:$LIBRARY_PATH

# For numpy setting to build with source code
COPY ./.numpy-site.cfg /root
COPY ./pip.conf /root/.config/pip/pip.conf

COPY ./requirements.txt /tmp
RUN pip3 install --upgrade pip setuptools \
    && pip3 install --no-binary :all: numpy \
    && pip3 install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# ffmpeg
RUN mkdir /ffmpeg-tmp \
    && wget -O /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz https://www.dropbox.com/s/n18np6ecgr8rqzg/ffmpeg-git-amd64-static.tar.xz \
    && tar xvf /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz -C /ffmpeg-tmp \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffmpeg /usr/local/bin \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffprobe /usr/local/bin \
    && rm -rf /ffmpeg-tmp \
    && apt-get clean \
    && rm -rf /var/chache/apt/archives/* /var/lib/apt/lists/*

WORKDIR /home
