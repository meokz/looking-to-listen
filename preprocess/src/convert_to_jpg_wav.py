import os
import sys
import glob
import math
import shlex
import argparse
import subprocess
from logging import getLogger, DEBUG

import cv2

import common.settings as settings
import env

logger = getLogger(__name__)
logger.setLevel(DEBUG)


def convert_to_jpg_wav(
        filepath: str, visual_dir: str, audio_dir: str, is_bulk: bool = False):

    cap = cv2.VideoCapture(filepath)
    frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Sometimes FPS appears at 0
    if fps == 0:
        return

    if env.mode == env.Mode.train:
        if is_bulk:
            sec = int(math.floor(frame / fps))
            iteraction_count = int(math.floor(sec / settings.DURATION))
        else:
            iteraction_count = 1
    else:
        sec = int(math.ceil(frame / fps))
        iteraction_count = int(math.ceil(sec / settings.DURATION))

    DEVNULL = open(os.devnull, 'wb')
    filename = os.path.basename(filepath).replace(".mp4", "")

    for i in range(iteraction_count):
        savepath_wav = audio_dir + "/{0}_{1}.wav".format(filename, i)
        if os.path.exists(savepath_wav):
            # already exists audio file
            continue

        savepath_vis = visual_dir + "/{0}_{1}".format(filename, i)
        if not os.path.exists(savepath_vis):
            os.makedirs(savepath_vis)

        start_time = i * settings.DURATION

        # 25 FPS
        cmd = 'ffmpeg -ss {0} -t {1} -r {2} -y {3} -i {4}'.format(
            start_time, settings.DURATION, settings.FPS, savepath_vis + "/%03d.jpg", filepath)
        popen1 = subprocess.Popen(shlex.split(cmd), stdout=DEVNULL, stderr=DEVNULL)

        # 160 kHZ
        cmd = 'ffmpeg -ss {0} -t {1} -r {2} -ar {3} -ac 1 -y {4} -i {5}'.format(
            start_time, settings.DURATION, settings.FPS, settings.SR, savepath_wav, filepath)
        popen2 = subprocess.Popen(shlex.split(cmd), stdout=DEVNULL, stderr=DEVNULL)

        popen1.wait()
        popen2.wait()


def main(args):

    if env.mode == env.Mode.train:
        INPUT_DIR = os.environ['AVSPEECH_DIR']
        OUTPUT_VIS_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/visual"
        OUTPUT_WAV_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/audio"
    else:
        INPUT_DIR = os.environ['MOVIE_DIR']
        OUTPUT_VIS_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/visual"
        OUTPUT_WAV_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/audio"

    if not os.path.exists(OUTPUT_VIS_DIR):
        os.makedirs(OUTPUT_VIS_DIR)

    if not os.path.exists(OUTPUT_WAV_DIR):
        os.makedirs(OUTPUT_WAV_DIR)

    logger.debug('loading data...')

    movies = sorted(glob.glob(INPUT_DIR + "/*.mp4"))

    to = len(movies) if args.to == -1 else args.to
    movies = movies[args.fr:to]

    logger.info(str(len(movies)) + " video will be process")

    for i, movie in enumerate(movies):
        convert_to_jpg_wav(
            filepath=movie,
            visual_dir=OUTPUT_VIS_DIR,
            audio_dir=OUTPUT_WAV_DIR,
            is_bulk=args.bulk,
        )
        sys.stdout.write("\r%d" % i)
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convert mp4 file to jpg and wav file'
    )
    parser.add_argument("--bulk", action='store_true', help="Bulk dataset generation")
    parser.add_argument("--fr", "-f", type=int, default=0, help="from")
    parser.add_argument("--to", "-t", type=int, default=-1, help="to")
    args = parser.parse_args()

    main(args)
