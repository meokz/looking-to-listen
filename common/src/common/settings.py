# ===== Movie ===== #
DURATION = 3    # seconds
FPS = 25        # framerate


# ===== Audio / STFT ===== #
SR = 16000      # Hz
FFT_SIZE = 512
HOP_LEN = 160   # 10 ms (10 msec * 16kHz = 160 frames)
WIN_LEN = 400   # 25 ms


# ===== Visual ===== #
IMAGE_SIZE = 220            # cropped image size
MINIMUM_IMAGE_SIZE = 160    # minimum image size
MARGIN = 44                 # cropped margin
FACTOR = 0.5                # resize factor
MISSING_NUM = 5             # missing face number
