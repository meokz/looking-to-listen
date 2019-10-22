import os
import sys
import glob
import shutil
import argparse
from logging import getLogger, DEBUG

import numpy as np
import pandas as pd


logger = getLogger(__name__)
logger.setLevel(DEBUG)


def random_audio(paths):

    length = len(paths)

    audio1 = np.random.randint(0, length)
    audio2 = np.random.randint(0, length)

    while audio1 == audio2:
        audio2 = np.random.randint(0, length)

    return paths[audio1], paths[audio2]


def generate_dataset(num, paths,
                     INPUT_FACE_DIR, OUTPUT_DIR, OUTPUT_WAV_DIR, OUTPUT_STFT_DIR):

    audio_stft1, audio_stft2 = random_audio(paths)

    face1 = INPUT_FACE_DIR + "/" + os.path.basename(audio_stft1)
    face2 = INPUT_FACE_DIR + "/" + os.path.basename(audio_stft2)

    if not os.path.exists(face1) or not os.path.exists(face2):
        # 顔写真がない
        return 0

    audio1 = np.load(audio_stft1)
    audio2 = np.load(audio_stft2)

    # TODO: There is bug. Pls see another script to combine multiple audio.
    # Convert waveform to F-T domain
    mix_stft = OUTPUT_STFT_DIR + "/{0}.npy".format(num)
    mixture = (audio1 + audio2) / 2.0
    np.save(mix_stft, mixture)

    mix_info = OUTPUT_DIR + "/" + str(num) + ".csv"
    df = pd.DataFrame([
        [mix_stft, audio_stft1, audio_stft2, face1, face2]],
        columns=['mix', 'clean1', 'clean2', 'visual1', 'visual2'])
    df.to_csv(mix_info, encoding="utf-8")

    return 1


def main(args):

    INPUT_AUDIO_DIR = os.environ['DATASET_DIR'] + "/avspeech/output/audio"
    INPUT_VISUAL_DIR = os.environ['DATASET_DIR'] + "/avspeech/output/visual"

    OUTPUT_DIR = os.environ['DATASET_DIR'] + "/2s/info"
    OUTPUT_WAV_DIR = os.environ['DATASET_DIR'] + "/2s/wav"
    OUTPUT_STFT_DIR = os.environ['DATASET_DIR'] + "/2s/stft"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_WAV_DIR):
        os.makedirs(OUTPUT_WAV_DIR)
    if not os.path.exists(OUTPUT_STFT_DIR):
        os.makedirs(OUTPUT_STFT_DIR)

    logger.debug("loading data...")
    audio_paths = sorted(glob.glob(INPUT_AUDIO_DIR + "/*.npy"))

    logger.info("generate synthesised sounds...")

    i = args.fr
    while i < args.num:
        result = generate_dataset(
            i, audio_paths,
            INPUT_VISUAL_DIR, OUTPUT_DIR, OUTPUT_WAV_DIR, OUTPUT_STFT_DIR)

        if result == 1:
            sys.stdout.write("\rNum: {0}".format(i))
            sys.stdout.flush()

        i += result

    sys.stdout.write("\n")

    logger.debug("remove wav files...")
    # 生成したWavファイルを削除する
    shutil.rmtree(OUTPUT_WAV_DIR + "/")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='generate dataset from dual face and noise.'
    )
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-num", type=int, default=132, help="to")
    args = parser.parse_args()

    main(args)
