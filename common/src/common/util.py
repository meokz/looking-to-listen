import librosa
import numpy as np

import common.settings as settings


def load_audio_and_stft(filepath: str):

    audio_wav, _ = librosa.load(filepath, sr=settings.SR)
    audio = librosa.stft(
        audio_wav, n_fft=settings.FFT_SIZE,
        hop_length=settings.HOP_LEN, win_length=settings.WIN_LEN)

    # 2-layer
    audio = np.stack([audio.real, audio.imag], 2)

    # 0埋め
    if audio.shape != (257, 301, 2):
        tmp = np.zeros((257, 301, 2))
        tmp[:audio.shape[0], :audio.shape[1], :] = audio
        audio = tmp
        print("reshape")

    return audio


def stft_and_save(input_path: str, output_path: str):

    try:
        audio_wav, _ = librosa.load(input_path, sr=settings.SR)
        audio = librosa.stft(
            audio_wav, n_fft=settings.FFT_SIZE,
            hop_length=settings.HOP_LEN, win_length=settings.WIN_LEN)
    except Exception as e:
        print(e)

    # 2-layer
    audio = np.stack([audio.real, audio.imag], 2)

    # 0埋め
    if audio.shape != (257, 301, 2):
        tmp = np.zeros((257, 301, 2))
        tmp[:audio.shape[0], :audio.shape[1], :] = audio
        audio = tmp
        print("reshape")

    np.save(output_path, audio)


def load_stft_and_norm(filepath):

    audio = np.load(filepath)
    audio = audio / np.max(np.abs(audio))
    return audio


def synthesis_two_audio_and_save(filepath1, filepath2, alpha, beta, savepath):

    from pydub import AudioSegment
    from pydub.utils import ratio_to_db

    audio1 = AudioSegment.from_wav(filepath1)
    audio2 = AudioSegment.from_wav(filepath2)

    audio1 = audio1 + ratio_to_db(alpha)
    audio2 = audio2 + ratio_to_db(beta)
    # synthesis_audio = audio1 + audio2
    synthesis_audio = audio1.overlay(audio2)

    # 'ffmpeg -ss {0} -t {1} -r {2} -ar {3} -ac 1 -y {4} -i {5}'
    synthesis_audio.export(
        savepath,
        format="wav",
        parameters=["-ar", str(settings.SR), "-ac", "1"]
    )


def istft_and_save(filepath, stft):

    signal = librosa.istft(stft, hop_length=settings.HOP_LEN, win_length=settings.FFT_SIZE)
    librosa.output.write_wav(path=filepath, y=signal, sr=settings.SR, norm=True)
