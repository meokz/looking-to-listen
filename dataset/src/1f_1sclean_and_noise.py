import os
import sys
import glob
import argparse
from logging import getLogger, DEBUG

import numpy as np
import pandas as pd

import common.util as util


logger = getLogger(__name__)
logger.setLevel(DEBUG)

INPUT_AUDIO_DIR = os.environ['DATASET_DIR'] + "/avspeech/audio/"
INPUT_VISUAL_DIR = os.environ['DATASET_DIR'] + "/avspeech/visual"
INPUT_NOISE_DIR = os.environ['DATASET_DIR'] + "/audioset/audio"

OUTPUT_INFO_DIR = os.environ['DATASET_DIR'] + "/1s_and_noise/info"
OUTPUT_MIX_DIR = os.environ['DATASET_DIR'] + "/1s_and_noise/mixture"

ALPHA = 0.7  # speech ratio
BETA = 0.3  # noise ratio


def random_audio(paths):

    length = len(paths)
    audio = np.random.randint(0, length)

    return paths[audio]


def generate_dataset(num, audio_paths, noise_paths):

    audio_stft = random_audio(audio_paths)
    noise_stft = random_audio(noise_paths)

    face = INPUT_VISUAL_DIR + "/" + os.path.basename(audio_stft)

    if not os.path.exists(face):
        # 顔動画がない
        return 0

    try:
        audio = util.load_stft_and_norm(audio_stft)
        noise = util.load_stft_and_norm(noise_stft)

        tmp1 = os.environ['DATASET_DIR'] + "/1s_and_noise/tmp1.wav"
        tmp2 = os.environ['DATASET_DIR'] + "/1s_and_noise/tmp2.wav"
        util.istft_and_save(tmp1, audio[:, :, 0] + audio[:, :, 1] * 1j)
        util.istft_and_save(tmp2, noise[:, :, 0] + noise[:, :, 1] * 1j)

        synthesis = os.environ['DATASET_DIR'] + "/1s_and_noise/tmp3.wav"
        util.synthesis_two_audio_and_save(tmp1, tmp2, ALPHA, BETA, synthesis)

        mixture = util.load_audio_and_stft(synthesis)
    except:
        import traceback
        print(traceback.format_exc())
        return 0

    mixture_path = OUTPUT_MIX_DIR + "/{0}.npy".format(num)
    np.save(mixture_path, mixture)

    mix_info = OUTPUT_INFO_DIR + "/{0}.csv".format(num)
    df = pd.DataFrame([
        [mixture_path, audio_stft, face]],
        columns=['mix', 'clean', 'visual'])
    df.to_csv(mix_info, encoding="utf-8")

    return 1


def main(args):

    if not os.path.exists(OUTPUT_INFO_DIR):
        os.makedirs(OUTPUT_INFO_DIR)

    if not os.path.exists(OUTPUT_MIX_DIR):
        os.makedirs(OUTPUT_MIX_DIR)

    logger.debug("loading data...")
    audio_paths = sorted(glob.glob(INPUT_AUDIO_DIR + "/*.npy"))
    noise_paths = sorted(glob.glob(INPUT_NOISE_DIR + "/*.npy"))

    logger.info("generate synthesised sounds...")

    i = args.fr
    while i < args.num:
        result = generate_dataset(i, audio_paths, noise_paths)

        if result == 1:
            sys.stdout.write("\rNum: {0}".format(i))
            sys.stdout.flush()

        i += result

    sys.stdout.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='generate dataset from single face and noise.')
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-n", "--num", type=int, default=132, help="to")
    args = parser.parse_args()

    main(args)
