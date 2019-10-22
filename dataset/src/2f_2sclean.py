import os
import sys
import argparse
from logging import getLogger, DEBUG

import numpy as np
import pandas as pd

import common.util as util


logger = getLogger(__name__)
logger.setLevel(DEBUG)

INPUT_AUDIO_DIR = os.environ['DATASET_DIR'] + "/avspeech/audio"
INPUT_VISUAL_DIR = os.environ['DATASET_DIR'] + "/avspeech/visual"

OUTPUT_INFO_DIR = os.environ['DATASET_DIR'] + "/2s/info"
OUTPUT_MIX_DIR = os.environ['DATASET_DIR'] + "/2s/mixture"


def random_audio(paths):

    length = len(paths)

    audio1 = np.random.randint(0, length)
    audio2 = np.random.randint(0, length)

    while audio1 == audio2:
        audio2 = np.random.randint(0, length)

    return paths[audio1], paths[audio2]


def generate_dataset(num, paths):

    audio_stft1, audio_stft2 = random_audio(paths)

    face1 = INPUT_VISUAL_DIR + "/" + os.path.basename(audio_stft1)
    face2 = INPUT_VISUAL_DIR + "/" + os.path.basename(audio_stft2)

    if not os.path.exists(face1) or not os.path.exists(face2):
        # no cropped face image
        return 0

    try:
        audio1 = util.load_stft_and_norm(audio_stft1)
        audio2 = util.load_stft_and_norm(audio_stft2)

        pid = os.getpid()
        tmp1 = os.environ['DATASET_DIR'] + "/2s/" + str(pid) + "_tmp1.wav"
        tmp2 = os.environ['DATASET_DIR'] + "/2s/" + str(pid) + "_tmp2.wav"

        util.istft_and_save(tmp1, audio1[:, :, 0] + audio1[:, :, 1] * 1j)
        util.istft_and_save(tmp2, audio2[:, :, 0] + audio2[:, :, 1] * 1j)

        synthesis = os.environ['DATASET_DIR'] + "/2s/" + str(pid) + "_tmp3.wav"
        util.synthesis_two_audio_and_save(tmp1, tmp2, 0.5, 0.5, synthesis)

        mixture = util.load_audio_and_stft(synthesis)
    except:
        import traceback
        print(traceback.format_exc())
        return 0

    mixture_path = OUTPUT_MIX_DIR + "/{0}.npy".format(num)
    np.save(mixture_path, mixture)

    mix_info = OUTPUT_INFO_DIR + "/{0}.csv".format(num)
    df = pd.DataFrame([
        [mixture_path, audio_stft1, audio_stft2, face1, face2]],
        columns=['mix', 'clean1', 'clean2', 'visual1', 'visual2'])
    df.to_csv(mix_info, encoding="utf-8")

    return 1


def main(args):

    if not os.path.exists(OUTPUT_INFO_DIR):
        os.makedirs(OUTPUT_INFO_DIR)

    if not os.path.exists(OUTPUT_MIX_DIR):
        os.makedirs(OUTPUT_MIX_DIR)

    logger.debug("loading data...")
    import glob
    audio_paths = sorted(glob.glob(INPUT_AUDIO_DIR + "/*.npy"))

    logger.info("generate synthesised sounds...")

    i = args.fr
    while i < args.num:
        result = generate_dataset(i, audio_paths)

        if result == 1:
            sys.stdout.write("\rNum: {0}".format(i))
            sys.stdout.flush()

        i += result

    sys.stdout.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='generate dataset for 2 clearn speacker.'
    )
    parser.add_argument("-f", "--fr", type=int, default=0, help="from")
    parser.add_argument("-n", "--num", type=int, default=132, help="to")
    args = parser.parse_args()

    main(args)
