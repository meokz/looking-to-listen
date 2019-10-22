# Generate Dataset

## Usage

* generate dataset with dual faces.

```sh
$ docker-compose run dataset python3 2f_2sclean.py -n [number]
```

if you want use another dataset (e.g. audio only, single face, combination of audioset), see other script.

* [0f_1sclean_and_noise.py](./src/0f_1sclean_and_noise.py) - no face (audio only), 1 clean speacker + noise from audioset
* [1f_1sclean_and_noise.py](./src/1f_1sclean_and_noise.py) - 1 face, 1 clean speacker + noise from audioset
* [2f_2sclean_and_noise.py](./src/2f_2sclean_and_noise.py) - 2 face, 2 clean speacker + noise from audioset
* [2s_2sclean.py](./src/2s_2sclean.py) - 2 face, 2 clean speacker
