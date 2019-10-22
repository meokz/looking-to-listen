import os
import wave
import math
import glob
import shlex
import shutil
import argparse
import subprocess

import chainer

import env
import common.util as util
import common.settings as settings
import modules.operation as op
from network.audio_only_net import Audio_Only_Net


xp = env.xp
model = None
ideep_mode = 'never'


def masking(audios):

    if env.gpu:
        audios = xp.asarray(audios)

    noise = xp.stack([audio.T for audio in audios]).astype(xp.float32)
    compressed_noise, _ = op.compress_audio(noise)

    print("estimate mask...")
    mask1, mask2 = model.estimate_mask(spec=compressed_noise)

    print("mul mask...")
    compressed_separated1 = op.mul(mask1, compressed_noise).data
    compressed_separated2 = op.mul(mask2, compressed_noise).data

    if env.gpu:
        compressed_separated1 = chainer.cuda.to_cpu(compressed_separated1)
        compressed_separated2 = chainer.cuda.to_cpu(compressed_separated2)

    print("reconstruct audio...")
    y1 = op.reconstruct_audio_complex(compressed_separated1)
    y2 = op.reconstruct_audio_complex(compressed_separated2)

    return y1, y2


def predict(audios):

    with chainer.using_config("train", False), \
            chainer.using_config('enable_backprop', False), \
            chainer.using_config('type_check', False), \
            chainer.using_config('use_ideep', ideep_mode):
        y1, y2 = masking(audios)

    return y1, y2


def noise_reduction(wav_path, save_path):

    save_dir = save_path + "/noise/segments"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    wf = wave.open(wav_path, "r")
    sec = float(wf.getnframes()) / wf.getframerate()
    wf.close()
    del wf

    segment_time = 3
    iteraction_count = int(math.ceil(sec / segment_time))

    for i in range(iteraction_count):
        start_time = i * segment_time
        path = save_dir + "/{0:02d}.wav".format(i)
        with open(os.devnull, 'wb') as devnull:
            cmd = 'ffmpeg -ss {0} -t {1} -ar {2} -ac 1 -y {3} -i {4}'.format(
                start_time, segment_time, settings.SR, path, wav_path)
            subprocess.call(shlex.split(cmd), stdout=devnull, stderr=devnull)

    segments = sorted(glob.glob(save_dir + "/*"))

    save_dir = save_path + "/noise/denoise_segments"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    audios = [util.load_audio_and_stft(segment) for segment in segments]
    y1, y2 = predict(audios)

    for i in range(y1.shape[2]):
        util.istft_and_save("{0}/{1:02d}.wav".format(save_dir, i), y1[:, :, i])

    segments = sorted(glob.glob(save_dir + "/*"))

    save_dir = save_path + "/noise"
    with open(save_dir + "/files.txt", "w") as f:
        for segement in segments:
            f.write("file \'" + segement + "\'\n")

    with open(os.devnull, 'wb') as devnull:
        cmd = 'ffmpeg -ac 1 -safe 0 -f concat -ss 0 -t {0} -y -i {1} {2}'.format(
            round(sec, 2),
            save_dir + "/files.txt",
            save_path + "/clean.wav")
        subprocess.call(shlex.split(cmd), stdout=devnull, stderr=devnull)


def main(args):
    global model, ideep_mode

    waves = sorted(glob.glob(args.audios + "/*.wav"))

    if not os.path.exists(os.environ['RESULT_DIR']):
        os.makedirs(os.environ['RESULT_DIR'])

    model = Audio_Only_Net()
    chainer.serializers.load_npz(args.model, model)

    if env.gpu:
        model.to_gpu(args.g)
    elif args.ideep:
        model.to_intel64()

    ideep_mode = 'always' if args.ideep else 'never'

    for wav in waves:
        basename = os.path.splitext(os.path.basename(wav))[0]
        save_path = os.environ['RESULT_DIR'] + "/" + basename

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print("wav:", wav)
        noise_reduction(wav, save_path)


if __name__ == "__main__":
    """ main endpoint """

    parser = argparse.ArgumentParser(
        description='description'
    )
    parser.add_argument("model", help="model or snapshot")
    parser.add_argument("audios", help="audio directory")
    parser.add_argument("--ideep", action='store_true', help="ideep (CPU Only)")
    parser.add_argument("-g", type=int, default=0, help="specify GPU")
    args = parser.parse_args()

    main(args)
