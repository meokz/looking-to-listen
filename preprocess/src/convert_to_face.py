import os
import sys
import time
import glob
import argparse
from logging import getLogger, DEBUG
from typing import Tuple

import scipy.misc
import numpy as np

import env
import common.settings as settings
from libs.facerec import FaceRec

logger = getLogger(__name__)
logger.setLevel(DEBUG)
facerec = FaceRec(gpu=0)


def crop(image, bounding_box, factor):

    if len(bounding_box) == 0:
        return np.zeros((settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3))

    # 1人目だけ考慮
    top, right, bottom, left = bounding_box[0]
    top = int(top * (1.0 / factor))
    right = int(right * (1.0 / factor))
    bottom = int(bottom * (1.0 / factor))
    left = int(left * (1.0 / factor))

    length = max(bottom - top, right - left) + settings.MARGIN
    offset_x = int((length - (bottom - top)) / 2.0)
    offset_y = int((length - (right - left)) / 2.0)
    x1 = max(top - offset_x, 0)
    x2 = min(bottom + offset_x, len(image[0]) - 1)
    y1 = max(left - offset_y, 0)
    y2 = min(right + offset_y, len(image[1]) - 1)

    # 破綻している場合
    if x1 > x2 or y1 > y2:
        return np.zeros((settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3))

    cropped = image[x1:x2, y1:y2, :]
    aligned = scipy.misc.imresize(
        cropped, (settings.IMAGE_SIZE, settings.IMAGE_SIZE), interp='bilinear')

    return aligned


def aliend_image(filepath: str, savepath: str) -> Tuple[int, int]:

    if os.path.exists(savepath + "/075.jpg"):
        # 既に生成されていたら処理を中断
        return 0, 0

    if os.path.exists(savepath + "/not_found.npy"):
        # NOTE: 既に処理済みだったら処理を中断
        return 0, 1

    # 各75枚の画像
    paths = sorted(glob.glob(filepath + "*.jpg"))

    if env.mode == env.Mode.train and len(paths) != 75:
        # もし75枚なかったら中断
        return 0, 1

    images = [scipy.misc.imread(path, mode='RGB') for path in paths]

    # リサイズ係数
    factor = settings.FACTOR
    small_images = [scipy.misc.imresize(image, (int(image.shape[0] * factor), int(
        image.shape[1] * factor)), interp='bilinear') for image in images]

    batch_of_face_locations = facerec.batch_face_locations(small_images, number_of_times_to_upsample=0, batch_size=75)

    missing_flames = [bounding_box for bounding_box in batch_of_face_locations if len(bounding_box) != 1]

    if env.mode == env.Mode.train and settings.MISSING_NUM < len(missing_flames):
        # 欠けているフレームもしくは
        # 2つ以上検出しているフレームが多かったら破棄
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        np.save(savepath + "/not_found", np.zeros(0))

        return 1, 1

    aligned_images = [crop(images[i], bounding_box, factor) for i, bounding_box in enumerate(batch_of_face_locations)]

    if env.mode == env.Mode.predict:
        # 実際のデータのとき、3秒以下の場合に0埋めをする
        while len(aligned_images) < 75:
            aligned_images.append(np.zeros((settings.IMAGE_SIZE, settings.IMAGE_SIZE, 3)))

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    [scipy.misc.imsave(savepath + "/{0:03d}.jpg".format(index + 1), image)
     for index, image in enumerate(aligned_images)]

    return 1, 0


def main(args):

    if env.mode == env.Mode.train:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/visual"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/face"
    else:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/visual"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/face"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.debug("loading data...")
    dirs = sorted(glob.glob(INPUT_DIR + "/**/"))
    to = len(dirs) if args.to == -1 else args.to
    dirs = dirs[args.fr:to]

    logger.info(str(len(dirs)) + " video will be process")

    total_time = 0
    total_missing = 0
    total_process = 0
    estimated_finish_time = 0

    logger.debug("convert jpg to face...")

    for i, dir in enumerate(dirs):

        start = time.time()

        savepath = "/" + dir.replace(INPUT_DIR, "").replace("/", "")

        # take 2~10 seconds
        process, missing = aliend_image(dir, OUTPUT_DIR + savepath)
        total_process += process
        total_missing += missing

        if process == 1:
            total_time += time.time() - start
            average_time = total_time / total_process
            remain_count = len(dirs) - i
            estimated_finish_time = round((average_time * remain_count) / 60, 2)

        total = i + 1
        sys.stdout.write(
            "\rfrom {0} to {1}- Sucess: {2} / {3}, Estimated Finish Time:{4} [min]".format(args.fr, to, total - total_missing, total, estimated_finish_time))
        sys.stdout.flush()

    sys.stdout.write("\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='convert image to cropped image.'
    )
    parser.add_argument("--fr", "-f", type=int, default=0, help="from")
    parser.add_argument("--to", "-t", type=int, default=-1, help="to")
    args = parser.parse_args()

    main(args)
