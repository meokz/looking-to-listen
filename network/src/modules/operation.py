import chainer
import numpy as np

import env
from env import xp


def mul(mask, compressed):

    if env.AUDIO_CHANNELS == 1:
        # simple mul
        return mask * compressed
    elif env.AUDIO_CHANNELS == 2:
        # complex_mul
        real = mask[:, 0, :, :] * compressed[:, 0, :, :] - mask[:, 1, :, :] * compressed[:, 1, :, :]
        imag = mask[:, 0, :, :] * compressed[:, 1, :, :] + compressed[:, 0, :, :] * mask[:, 1, :, :]

        import chainer.functions as F
        result = F.stack([real, imag], axis=1)
        return result


def compress_audio(audio):

    # NOTE: 入力が正規化されているかどうか未知であるため
    # 必ず先頭でNormalizationする
    norm = xp.stack([i / xp.max(xp.abs(i)) for i in audio])

    mag = audio[:, 0, :, :] ** 2 + audio[:, 1, :, :] ** 2
    mag = xp.stack([i / xp.max(i) for i in mag])

    # power-low compression
    compressed = None
    if env.AUDIO_CHANNELS == 1:
        compressed = (mag ** 0.3)[:, xp.newaxis, :, :]
    elif env.AUDIO_CHANNELS == 2:
        compressed = (xp.abs(norm) ** 0.3) * xp.sign(audio)

    return compressed, mag


def reconstruct_audio(compressed, reference):

    power = xp.sqrt(xp.power(compressed, 1 / 0.3))[:, 0, :, :]
    power = chainer.cuda.to_cpu(power).T

    reference = chainer.cuda.to_cpu(reference).T
    reference = reference[:, :, 0, :] + reference[:, :, 1, :] * 1j
    phase = np.exp(1.0j * np.angle(reference))

    stft = power * phase

    return stft


def reconstruct_audio_complex(compressed):

    compressed = compressed.T
    compressed = (np.abs(compressed) ** (1 / 0.3)) * np.sign(compressed)

    real = compressed[:, :, 0, :]
    imag = compressed[:, :, 1, :]

    stft = real + imag * 1j

    return stft
