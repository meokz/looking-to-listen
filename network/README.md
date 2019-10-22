# Train and Predict

## Open in bash

```sh
$ docker-compose run network bash
```

## Setting network parameter

see `network/src/env.py`

## Train (GPU Only)

single gpu
```sh
$ python3 train.py
```

multiple gpu
```sh
$ python3 train.py -g0 0 -g1 1 -g2 2 -g3 3
```

## Predict

See `--help` for intel cpu optimization and gpu options.

```sh
$ python3 predict_from_dataset.py
```

```sh
$ python3 predict.py
```
