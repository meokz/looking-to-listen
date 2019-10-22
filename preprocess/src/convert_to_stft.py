import os
import sys
import glob
from logging import getLogger, DEBUG

import common.util as util
import env

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def stft(filepath, OUTPUT_DIR):

    save_path = OUTPUT_DIR + "/" + os.path.basename(filepath).replace(".wav", ".npy")
    util.stft_and_save(filepath, save_path)


def main(args):

    if env.mode == env.Mode.train:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/audio"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/audio"
    else:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/audio"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/movie/audio"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.debug("loading data...")
    audio_paths = sorted(glob.glob(INPUT_DIR + "/*.wav"))
    to = len(audio_paths) if args.to == -1 else args.to
    audio_paths = audio_paths[args.fr:to]

    logger.info(str(len(audio_paths)) + " audio will be process")

    logger.debug("convert wavefile to Time-Frequency domain by stft...")
    for i, path in enumerate(audio_paths):
        stft(path, OUTPUT_DIR)
        sys.stdout.write("\r{0}".format(i))
        sys.stdout.flush()

    sys.stdout.write("\n")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description='convert to T-F Domain from wav file.'
    )
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-to", type=int, default=-1, help="to")
    args = parser.parse_args()

    main(args)
