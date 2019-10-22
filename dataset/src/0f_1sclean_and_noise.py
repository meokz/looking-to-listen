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

INPUT_AUDIO_DIR = os.environ['DATASET_DIR'] + "/avspeech/audio"
INPUT_NOISE_DIR = os.environ['DATASET_DIR'] + "/audioset/audio"

OUTPUT_INFO_DIR = os.environ['DATASET_DIR'] + "/0f_1sclean_noise/info"
OUTPUT_MIX_DIR = os.environ['DATASET_DIR'] + "/0f_1sclean_noise/mixture"

labels = pd.read_csv(
    os.environ['DATASET_DIR'] + '/audioset/unbalanced_train_segments.csv',
    names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
    sep=', ',
    quotechar='"')

speech_label = [
    "/m/09x0r",   # "Speech"
    "/m/05zppz",  # "Male speech, man speaking"
    "/m/02zsn",   # "Female speech, woman speaking"
    "/m/0ytgt",   # "Child speech, kid speaking"
    "/m/01h8n0",  # "Conversation"
    "/m/02qldy",  # "Narration, monologue"
    "/m/0261r1",  # "Babbling"
    "/m/0brhx",   # "Speech synthesizer",
    "/m/02rtxlg",  # "Whispering"
]


def random_audio(paths):

    length = len(paths)
    audio = np.random.randint(0, length)

    return paths[audio]


def get_noise_label(filepath):

    filename = os.path.basename(filepath)

    youtube_id = filename[:11]
    # info = filename[11:]
    # start_time = info.split("_")[1]
    # end_time = info.split("_")[2]

    label = labels.query('YTID == "' + youtube_id + '"')

    if len(label.index) == 0:
        return None

    if len(label.index) > 1:
        # TODO;
        return None

    for _, row in label.iterrows():
        positive_labels = row["positive_labels"].replace('"', '').split(",")
        return positive_labels


def check_noise(positive_labels):

    for label in positive_labels:
        if label in speech_label:
            return False

    return True


def get_noise(noises):

    is_noise = False

    noise = None
    while not is_noise:
        noise = random_audio(noises)

        positive_labels = get_noise_label(noise)
        if positive_labels is None:
            continue

        is_noise = check_noise(positive_labels)

    return noise


def get_ratio():

    noise_ratio = np.random.randint(0, 6) / 10.0
    audio_ratio = 1.0 - noise_ratio
    return audio_ratio, noise_ratio


def generate_dataset(num, audio_paths, noise_paths):

    audio_stft = random_audio(audio_paths)
    noise_stft = get_noise(noise_paths)

    try:
        audio = util.load_stft_and_norm(audio_stft)
        noise = util.load_stft_and_norm(noise_stft)

        pid = os.getpid()
        tmp1 = os.environ['DATASET_DIR'] + "/0f_1sclean_noise/" + str(pid) + "_tmp1.wav"
        tmp2 = os.environ['DATASET_DIR'] + "/0f_1sclean_noise/" + str(pid) + "_tmp2.wav"
        util.istft_and_save(tmp1, audio[:, :, 0] + audio[:, :, 1] * 1j)
        util.istft_and_save(tmp2, noise[:, :, 0] + noise[:, :, 1] * 1j)

        synthesis = os.environ['DATASET_DIR'] + "/0f_1sclean_noise/" + str(pid) + "_tmp3.wav"
        audio_ratio, noise_ratio = get_ratio()
        util.synthesis_two_audio_and_save(tmp1, tmp2, audio_ratio, noise_ratio, synthesis)

        mixture = util.load_audio_and_stft(synthesis)
    except:
        import traceback
        print(traceback.format_exc())
        return 0

    mixture_path = OUTPUT_MIX_DIR + "/{0}.npy".format(num)
    np.save(mixture_path, mixture)

    mix_info = OUTPUT_INFO_DIR + "/{0}.csv".format(num)
    df = pd.DataFrame([[mixture_path, audio_stft]], columns=['mix', 'clean'])
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
        description='generate dataset from clean speacker and noise.')
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-n", "--num", type=int, default=132, help="to")
    args = parser.parse_args()
    main(args)
