import os

import chainer
import numpy as np
import pandas as pd


class AudioOnlyDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):

        info = pd.read_csv(
            os.environ['DATASET_DIR'] + "/0f_1sclean_noise/info/" + "/{0}.csv".format(i))

        noise = np.load(info['mix'][0]).T.astype(np.float32)
        clean = np.load(info['clean'][0]).T.astype(np.float32)

        return noise, clean


class SingleFaceDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):

        info = pd.read_csv(
            os.environ['DATASET_DIR'] + "/1s_and_noise/info/" + "/{0}.csv".format(i))

        noise = np.load(info['mix'][0]).T.astype(np.float32)
        clean = np.load(info['clean'][0]).T.astype(np.float32)
        visual = np.load(info['visual'][0]).T.astype(np.float32)

        return noise, clean, visual


class DualFaceDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):

        info = pd.read_csv(
            os.environ['DATASET_DIR'] + "/2s/info/" + "/{0}.csv".format(i))

        noise = np.load(info['mix'][0]).T.astype(np.float32)
        clean1 = np.load(info['clean1'][0]).T.astype(np.float32)
        visual1 = np.load(info['visual1'][0]).T.astype(np.float32)
        clean2 = np.load(info['clean2'][0]).T.astype(np.float32)
        visual2 = np.load(info['visual2'][0]).T.astype(np.float32)

        return noise, clean1, clean2, visual1, visual2
