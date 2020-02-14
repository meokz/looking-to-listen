import numpy as np
xp = np
gpu = False


def _set_cpu():
    global xp, gpu
    xp = np
    gpu = False


def _set_gpu():
    global xp, gpu
    import chainer
    xp = chainer.cuda.cupy
    gpu = True


# ===== CPU or GPU ===== #
# _set_cpu()
_set_gpu()

# ===== Network ===== #
# model filename
MODEL_NAME = "2f_2sclean_fc7"
# input face number
INPUT_FACE = 0
# output mask number
OUTPUT_MASK = 1
# mag (only power) :1, complex: 2
AUDIO_CHANNELS = 2
VIS_CHANNNEL = 1792
AUDIO_LEN = 301
# Size of Fully Connected layer
FC_ROW = 1


# ===== Training ===== #
BATCH_SIZE = 6
ITERATION = 5000000
TRAIN = 1500000
EVALUATION = 10


def print_settings():

    print('==========================================')
    print('Input Face Length:', INPUT_FACE)
    print('Output Mask Length:', OUTPUT_MASK)
    print('Audio channels:', AUDIO_CHANNELS)
    print('Iteration:', ITERATION)
    print('Batch size:', BATCH_SIZE)
    print('Train dataset Size:', TRAIN)
    print('Epoch:', (ITERATION * BATCH_SIZE) / TRAIN)
    print('Evaluation dataset Size:', EVALUATION)
    print('==========================================')
