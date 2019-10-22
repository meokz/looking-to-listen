FROM nvidia/cuda:9.0-cudnn7-devel

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends  \
    wget \
    curl \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/chache/apt/archives/* /var/lib/apt/lists/*

# ffmpeg
RUN mkdir /ffmpeg-tmp \
    && wget -O /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz https://www.dropbox.com/s/n18np6ecgr8rqzg/ffmpeg-git-amd64-static.tar.xz \
    && tar xvf /ffmpeg-tmp/ffmpeg-git-amd64-static.tar.xz -C /ffmpeg-tmp \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffmpeg /usr/local/bin \
    && cp /ffmpeg-tmp/ffmpeg-git-20190612-amd64-static/ffprobe /usr/local/bin \
    && rm -rf /ffmpeg-tmp

# pip
RUN mkdir /tmp-pip
COPY ./requirements.txt /tmp-pip
RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3 \
    && pip3 install --upgrade pip setuptools \
    && pip3 install -r /tmp-pip/requirements.txt \
    && rm -rf /tmp-pip
