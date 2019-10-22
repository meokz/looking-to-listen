import chainer
import chainer.functions as F

from network.stream.audio_stream import Audio_Stream
from network.stream.visual_stream import Visual_Stream
from network.stream.fusion_stream import Fusion_Stream
from network.loss import evaluate_loss

import modules.operation as op
import env
from env import xp


class Single_Complex_Net(chainer.Chain):

    def __call__(self, noise, clean, face):

        noise = xp.asarray(noise).astype(xp.float32)
        clean = xp.asarray(clean).astype(xp.float32)
        face = xp.asarray(face).astype(xp.float32)[:, :, :, xp.newaxis]

        compressed_noise, _ = op.compress_audio(noise)
        compressed_clean, _ = op.compress_audio(clean)

        mask, _ = self.estimate_mask(spec=compressed_noise, face=face)

        separated = op.mul(mask, compressed_noise)

        loss = evaluate_loss(self, separated, compressed_clean)

        return loss

    def estimate_mask(self, spec, face):

        # ===== Audio Stream ===== #
        a = self.audio_stream(spec)
        a = F.concat([a[:, i, :, :] for i in range(a.shape[1])], axis=2)  # (b, 301, 2056)
        a = F.transpose(a, (0, 2, 1))[:, :, :, xp.newaxis]  # (b, 2056, 301, 1)

        # ===== Visual Streams ===== #
        b = self.visual_stream(face)
        x = b

        # ===== Fusion Stream ===== #
        x = F.concat((x, a), axis=1)   # (b, 2568, 301, 1)
        x = F.transpose(x, (0, 2, 1, 3))[:, :, :, 0]  # (b, 301, 2568)

        y = self.fusion_stream(x)

        return y[:, 0, :, :, :], 1 - y[:, 0, :, :, :]

    def __init__(self, trained=None):

        super(Single_Complex_Net, self).__init__()

        with self.init_scope():

            if env.AUDIO_LEN % env.FC_ROW != 0:
                print("invalid fc layer parameter")
                import sys
                sys.exit(1)

            # ===== Audio Stream ===== #
            self.audio_stream = Audio_Stream(
                trained=None if trained is None else trained["audio_stream"]
            )

            # ===== Visual Streams ===== #
            self.visual_stream = Visual_Stream(
                trained=None if trained is None else trained["visual_stream"]
            )
            self.visual_stream.disable_update()

            # ===== Fusion Stream ===== #
            self.fusion_stream = Fusion_Stream(
                trained=None if trained is None else trained["fusion_stream"]
                # trained=None if trained is None else trained
            )
