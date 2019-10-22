import os
import sys
import glob
from logging import getLogger, DEBUG

import numpy as np
from scipy import misc
import tensorflow as tf

import env
import libs.facenet as facenet


FACENET_MODEL = "/model/20180402-114759.pb"
logger = getLogger(__name__)
logger.setLevel(DEBUG)


def convert_to_vector(face, sess, input_dir, output_dir):

    filename = face.replace(input_dir, "").replace("/", "")
    savepath = output_dir + "/" + filename + ".npy"

    if os.path.exists(savepath):
        # 既に変換済みだったら抜ける
        return

    paths = sorted(glob.glob(face + "/*.jpg"))

    if len(paths) != 75:
        # 顔検出されなていなかったら抜ける
        return

    images = [misc.imread(path, mode='RGB') for path in paths]
    images = [facenet.prewhiten(image) for image in images]
    images = np.stack(images)

    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name(
        "InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # Run forward pass to calculate embeddings
    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    emb = sess.run(embeddings, feed_dict=feed_dict)

    # Save
    np.save(savepath, emb[:, 0, 0])


def main(args):

    if env.mode == env.Mode.train:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/mediate/face"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/avspeech/visual"
    else:
        INPUT_DIR = os.environ['DATASET_DIR'] + "/movie/mediate/face"
        OUTPUT_DIR = os.environ['DATASET_DIR'] + "/movie/visual"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    logger.debug('loading data...')
    faces = sorted(glob.glob(INPUT_DIR + "/**/"))
    to = len(faces) if args.to == -1 else args.to
    faces = faces[args.fr:to]

    logger.info(str(len(faces)) + " video will be process")

    with tf.Graph().as_default():

        config = tf.ConfigProto()
        config.log_device_placement = False
        config.gpu_options.visible_device_list = "0"
        # True->必要になったら確保, False->最初から全部確保
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)

        with tf.Session() as sess:

            logger.debug("loading model...")
            facenet.load_model(FACENET_MODEL)

            logger.debug("start process...")
            for i, face in enumerate(faces):
                convert_to_vector(
                    face=face,
                    sess=sess,
                    input_dir=INPUT_DIR,
                    output_dir=OUTPUT_DIR
                )

                sys.stdout.write(
                    "\rfrom {0} to {1}- {2}".format(args.fr, to, i))
                sys.stdout.flush()

            sys.stdout.write("\n")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
        description='Convert cropped jpg to face vector.'
    )
    parser.add_argument("-fr", type=int, default=0, help="from")
    parser.add_argument("-to", type=int, default=-1, help="to")
    args = parser.parse_args()

    main(args)
