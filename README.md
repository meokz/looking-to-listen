# Looking to Listen

This is implementation of ["Looking to Listen at the Cocktail Party"](https://looking-to-listen.github.io/) by python3 and chainer.
This deep learning technology can be applied to noise reduction, removal of background music, and speech separation.

Original paper is [here](https://looking-to-listen.github.io/) (arxiv.org/abs/1804.03619).
Note that this implementation is inspired by [crystal-method](https://github.com/crystal-method/Looking-to-Listen) (MIT).

## Quick Start Demonstration (Audio-only Noise Reduction)

We show demonstration of noise reduction using pretrained model.

1.  First, you need build docker container.
```sh
$ docker-compose build
```

2. Put the noisy audio file(s) to `./data/noise`.

3. Run following command.

* GPU
```sh
$ docker-compose run network python3 quick_start_audio_only.py /data/model/0f_1sclean_noise.npz /data/noise
```

* CPU (comment out `_set_gpu()` in `network/src/env.py`)
```sh
Intel CPU (Fast)
$ docker-compose run network python3 quick_start_audio_only.py /data/model/0f_1sclean_noise.npz /data/noise -ideep
Other CPU (Slow)
$ docker-compose run network python3 quick_start_audio_only.py /data/model/0f_1sclean_noise.npz /data/noise
```

4. We can get clean audio in `./data/results`.

## Usage

Please refer to the following section for additional information such as speech separation and audio-visual processing.

* [Data Preprocess](./preprocess)
* [Generate Dataset](./dataset)
* [Train and Predict](./network)

### Open in bash

```sh
$ docker-compose run preprocess bash
```

```sh
$ docker-compose run dataset bash
```

```sh
$ docker-compose run network bash
```

## Differences from original paper

The original paper has a large FC layer.
However, there is not enough memory to put this network on the GPU.
In this implementation, the size of the FC layer is reduced so that a network can be installed in a single GPU.

## External Libraries

We use external libraries in `preprocess/src/libs`.

* [Facenet](https://github.com/davidsandberg/facenet) (MIT)
