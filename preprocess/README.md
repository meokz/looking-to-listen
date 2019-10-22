# Data Preprocess

Before train or predict, we need a pre-process to convert the data into feature vectors.

## Generating training data

### AVSpeech

1. Put mp4 files in `AVSPEECH_DIR`.

2. Run (GPU Only)
```sh
$ docker-compose run preprocess ./run.sh
```

### AudioSet

1. Put mp4 files in `AUDIOSET_DIR`.

2. Run
```sh
$ docker-compose run preprocess python3 convert_audioset.py
```

## Generating predicting data

1. Set `mode = Mode.predict` appeared in `preprocess/src/env.py`.

2. Put mp4 files in `MOVIE_DIR`.

3. Run (GPU Only)
```sh
$ docker-compose run preprocess ./run.sh
```
