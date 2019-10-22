import os
import sys
import math
import argparse
import traceback
from logging import getLogger, DEBUG

from pydub import AudioSegment

import common.settings as settings
import common.util as util


logger = getLogger(__name__)
logger.setLevel(DEBUG)


def convert_to_stft(filepath: str, wav_dir: str, stft_dir: str, is_bulk: bool):

    filename = os.path.basename(filepath).replace(".flac", "")

    audio = AudioSegment.from_file(filepath)
    audio.set_channels(1)

    iteraction_count = int(math.floor(audio.duration_seconds / settings.DURATION))

    if not is_bulk:
        iteraction_count = 1

    for i in range(iteraction_count):
        savepath_wav = wav_dir + "/{0}_{1}.wav".format(filename, i)

        start_time = (i * settings.DURATION) * 1000
        end_time = start_time + settings.DURATION * 1000

        _audio = audio[start_time:end_time]
        _audio.export(
            savepath_wav,
            format="wav",
            parameters=[
                "-ar", settings.SR, "-ac", "1"
            ])

        savepath_stft = stft_dir + "/{0}_{1}.npy".format(filename, i)
        util.stft_and_save(input_path=savepath_wav, output_path=savepath_stft)
        os.remove(savepath_wav)


def main(args):

    INPUT_DIR = os.environ['AUDIOSET_DIR']
    OUTPUT_WAV_DIR = os.environ['DATASET_DIR'] + "/audioset/mediate/wav"
    OUTPUT_STFT_DIR = os.environ['DATASET_DIR'] + "/audioset/audio"

    if not os.path.exists(OUTPUT_WAV_DIR):
        os.makedirs(OUTPUT_WAV_DIR)

    if not os.path.exists(OUTPUT_STFT_DIR):
        os.makedirs(OUTPUT_STFT_DIR)

    logger.debug('loading data...')

    import glob
    audios = sorted(glob.glob(INPUT_DIR + "/*.flac"))
    to = len(audios) if args.to == -1 else args.to
    audios = audios[args.fr:to]

    logger.info(str(len(audios)) + " audio file will be process")

    for i, audio in enumerate(audios):
        try:
            convert_to_stft(
                filepath=audio,
                wav_dir=OUTPUT_WAV_DIR,
                stft_dir=OUTPUT_STFT_DIR,
                is_bulk=args.bulk
            )
        except:
            logger.error(traceback.format_exc())

        sys.stdout.write("\r%d" % i)
        sys.stdout.flush()

    sys.stdout.write("\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='convert to T-F Fomain from wav.'
    )
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-to", type=int, default=-1, help="to")
    parser.add_argument("--bulk", action='store_true', help="Bulk dataset")
    args = parser.parse_args()

    main(args)
