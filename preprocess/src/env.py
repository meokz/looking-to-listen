import enum


class Mode(enum.Enum):
    train = 0
    predict = 1


# ==== Preprocess mode ==== #
mode = Mode.predict
