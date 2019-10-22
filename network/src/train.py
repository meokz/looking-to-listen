import os
import glob
import argparse

import chainer
from chainer import cuda
from chainer.training import extensions

import env
from network.audio_only_net import Audio_Only_Net
from network.single_complex_net import Single_Complex_Net
from network.double_complex_net import Double_Complex_Net
from network.preprocess import AudioOnlyDataset, SingleFaceDataset, DualFaceDataset


def get_dataset(dataset_name):

    dataset = glob.glob(os.path.join(os.environ['DATASET_DIR'] + "/" + dataset_name + "/info", "*.csv"))

    if len(dataset) < env.TRAIN + env.EVALUATION:
        print("ERROR: dataset size is small. len(dataset) =", len(dataset))

    all_nums = range(len(dataset))
    train = all_nums[:env.TRAIN]
    test = all_nums[env.TRAIN:env.TRAIN + env.EVALUATION]

    return train, test


def main(args):

    if not os.path.exists(os.environ['MODEL_DIR']):
        os.makedirs(os.environ['MODEL_DIR'])

    env.print_settings()
    print('Using GPUs:', args.gpu0, args.gpu1, args.gpu2, args.gpu3)

    if args.gpu1 == -1:
        cuda.get_device(args.gpu0).use()

    chainer.backends.cuda.get_device_from_id(args.gpu0).use()

    print("loading model and dataset...")
    trained = None

    if env.INPUT_FACE == 1:
        model = Single_Complex_Net(trained)
        train, test = get_dataset("1s_and_noise")
        train_data = SingleFaceDataset(train)
        val_data = SingleFaceDataset(test)
    elif env.INPUT_FACE == 2:
        model = Double_Complex_Net(trained)
        train, test = get_dataset("2s")
        train_data = DualFaceDataset(train)
        val_data = DualFaceDataset(test)
    else:
        model = Audio_Only_Net(trained)
        train, test = get_dataset("0f_1sclean_noise")
        train_data = AudioOnlyDataset(train)
        val_data = AudioOnlyDataset(test)

    if args.gpu1 == -1:
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(alpha=3 * 1e-5)
    optimizer.setup(model)

    # NOTE: Multiprocessを使うと謎警告が出る
    import warnings
    warnings.filterwarnings('ignore')
    train_iter = chainer.iterators.MultiprocessIterator(
        dataset=train_data, batch_size=env.BATCH_SIZE,
        shuffle=True, repeat=True)
    test_iter = chainer.iterators.MultiprocessIterator(
        dataset=val_data, batch_size=env.BATCH_SIZE,
        shuffle=False, repeat=False)

    print("setting trainer...")

    if args.gpu3 != -1:
        updater = chainer.training.ParallelUpdater(
            # updater = chainer.training.MultiprocessParallelUpdater(
            train_iter, optimizer,
            devices={
                'main': args.gpu0, 'second': args.gpu1, 'third': args.gpu2, 'fourth': args.gpu3
            },
        )
    elif args.gpu2 != -1:
        updater = chainer.training.ParallelUpdater(
            # updater = chainer.training.MultiprocessParallelUpdater(
            train_iter, optimizer,
            devices={
                'main': args.gpu0, 'second': args.gpu1, 'third': args.gpu2,
            },
        )
    elif args.gpu1 != -1:
        updater = chainer.training.ParallelUpdater(
            # updater = chainer.training.MultiprocessParallelUpdater(
            train_iter, optimizer,
            devices={
                'main': args.gpu0, 'second': args.gpu1
            },
        )
    else:
        updater = chainer.training.updaters.StandardUpdater(
            train_iter, optimizer, device=args.gpu0)

    trainer = chainer.training.Trainer(updater, (env.ITERATION, "iteration"), out=os.environ['MODEL_DIR'])
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu0), trigger=(100, "iteration"))
    trainer.extend(extensions.LogReport(trigger=(100, "iteration"), log_name=env.MODEL_NAME + "_log"))
    trainer.extend(extensions.ProgressBar(update_interval=1))
    trainer.extend(
        extensions.PlotReport(
            ["main/loss", "validation/main/loss"], "iteration",
            file_name=env.MODEL_NAME + "_loss.png",
            trigger=(100, "iteration")
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ["epoch", "iteration", "main/loss", "validation/main/loss", "elapsed_time"]
        ),
        trigger=(100, "iteration")
    )
    trainer.extend(
        extensions.snapshot(
            filename=env.MODEL_NAME + '_snapshot_iter_{.updater.iteration}'),
        trigger=(3000, "iteration")
    )

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    print("start training...")
    chainer.config.autotune = True
    trainer.run()

    print("saving model...")
    model.to_cpu()

    chainer.serializers.save_npz(
        os.environ['MODEL_DIR'] + "/" + env.MODEL_NAME + "_model.npz", model)
    chainer.serializers.save_npz(
        os.environ['MODEL_DIR'] + "/" + env.MODEL_NAME + "_optimizer.npz", optimizer)

    print("done!!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0", "-g0", type=int, default=0, help="Main GPU")
    parser.add_argument("--gpu1", "-g1", type=int, default=-1, help="Second GPU")
    parser.add_argument("--gpu2", "-g2", type=int, default=-1, help="Third GPU")
    parser.add_argument("--gpu3", "-g3", type=int, default=-1, help="Fourth GPU")
    parser.add_argument("--resume", default="", help="Resume the training from snapshot")
    args = parser.parse_args()

    main(args)
