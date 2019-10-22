import os

import pandas as pd

from env import xp


def load_dataset_audio(num):

    info = [pd.read_csv(
        os.environ['DATASET_DIR'] + "/0f_1sclean_noise/info" + "/{0}.csv".format(i)) for i in num]

    # 'mix', 'clean1', 'clean2', 'visual1', 'visual2'
    noise = xp.stack([xp.load(i['mix'][0]).T for i in info]).astype(xp.float32)

    clean = xp.stack([xp.load(i['clean'][0]).T for i in info]).astype(xp.float32)

    return noise, clean


def load_dataset_single(num):

    info = [pd.read_csv(
        os.environ['DATASET_DIR'] + "/1s_and_noise/info" + "/{0}.csv".format(i)) for i in num]

    # 'mix', 'clean1', 'clean2', 'visual1', 'visual2'
    noise = xp.stack([xp.load(i['mix'][0]).T for i in info]).astype(xp.float32)

    clean = xp.stack([xp.load(i['clean'][0]).T for i in info]).astype(xp.float32)
    visual = xp.stack([xp.load(i['visual'][0]).T for i in info]).astype(xp.float32)
    visual = visual[:, :, :, xp.newaxis]

    return noise, clean, visual


def load_dataset_double(num):

    info = [pd.read_csv(
        os.environ['DATASET_DIR'] + "/2s/info" + "/{0}.csv".format(i)) for i in num]

    # 'mix', 'clean1', 'clean2', 'visual1', 'visual2'
    noise = xp.stack([xp.load(i['mix'][0]).T for i in info]).astype(xp.float32)

    clean1 = xp.stack([xp.load(i['clean1'][0]).T for i in info]).astype(xp.float32)
    visual1 = xp.stack([xp.load(i['visual1'][0]).T for i in info]).astype(xp.float32)
    visual1 = visual1[:, :, :, xp.newaxis]

    clean2 = xp.stack([xp.load(i['clean2'][0]).T for i in info]).astype(xp.float32)
    visual2 = xp.stack([xp.load(i['visual2'][0]).T for i in info]).astype(xp.float32)
    visual2 = visual2[:, :, :, xp.newaxis]

    return noise, clean1, clean2, visual1, visual2
