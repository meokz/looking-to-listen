FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends  \
    git \
    gcc \
    make \
    curl \
	wget \
    cmake \
    unzip \
    libsm6 \
    libxext6 \
    libpq-dev \
    libxrender1 \
    libglib2.0-0 \
    liblapack-dev \
    libxrender-dev \
    libopenblas-dev \
    build-essential \
    ca-certificates \
    python3-dev \
    && curl -kL https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install --upgrade pip setuptools \
    && pip3 install scikit-build

# Install compilers
RUN apt install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt update -y \
    && apt install -y \
    gcc-6 \
    g++-6 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-6 50

# Install dlib
RUN git clone -b 'v19.16' --single-branch https://github.com/davisking/dlib.git \
    && mkdir -p /dlib/build \
    && cmake -H/dlib -B/dlib/build -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 \
    && cmake --build /dlib/build \
    && cd /dlib; python3 /dlib/setup.py install

# pip
RUN mkdir /tmp-pip
COPY ./requirements.txt /tmp-pip
RUN pip3 install -r /tmp-pip/requirements.txt \
    && rm -rf /tmp-pip

# ffmpeg
RUN mkdir /ffmpeg-tmp \
    && wget -O /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz https://www.dropbox.com/s/n18np6ecgr8rqzg/ffmpeg-git-amd64-static.tar.xz \
    && tar xvf /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz -C /ffmpeg-tmp \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffmpeg /usr/local/bin \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffprobe /usr/local/bin \
    && rm -rf /ffmpeg-tmp

# download facenet model
RUN mkdir /model \
    && mkdir /model-tmp \
    && cd /model-tmp \
    && curl gdrive.sh | bash -s https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view?usp=sharing \
    && unzip 20180402-114759.zip \
    && mv 20180402-114759/20180402-114759.pb /model \
    && cd /home \
    && rm -rf /model-tmp \
    && apt-get clean \
    && rm -rf /var/chache/apt/archives/* /var/lib/apt/lists/*
