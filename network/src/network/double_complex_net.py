import chainer
import chainer.functions as F

from network.stream.audio_stream import Audio_Stream
from network.stream.visual_stream import Visual_Stream
from network.stream.fusion_stream import Fusion_Stream
from network.loss import evaluate_loss

import modules.operation as op
import env as env
from env import xp


class Double_Complex_Net(chainer.Chain):

    def __call__(self, noise, clean1, clean2, face1, face2):

        noise = xp.asarray(noise).astype(xp.float32)
        clean1 = xp.asarray(clean1).astype(xp.float32)
        clean2 = xp.asarray(clean2).astype(xp.float32)
        face1 = xp.asarray(face1).astype(xp.float32)[:, :, :, xp.newaxis]
        face2 = xp.asarray(face2).astype(xp.float32)[:, :, :, xp.newaxis]

        clean = xp.concatenate((clean1, clean2), axis=3)

        compressed_noise, _ = op.compress_audio(noise)
        compressed_clean, _ = op.compress_audio(clean)

        mask1, mask2 = self.estimate_mask(
            spec=compressed_noise, face1=face1, face2=face2)

        separated1 = op.mul(mask1, compressed_noise)
        separated2 = op.mul(mask2, compressed_noise)

        separated = F.concat((separated1, separated2), axis=3)  # (6, 2, 301, 514)

        loss = evaluate_loss(self, separated, compressed_clean)
        return loss

    def estimate_mask(self, spec, face1, face2=None):

        # ===== Audio Stream ===== #
        a = self.audio_stream(spec)

        # ===== Visual Streams ===== #
        b = self.visual_stream(face1)
        c = self.visual_stream(face2)

        # ===== Fusion Stream ===== #
        a = F.concat([a[:, i, :, :] for i in range(a.shape[1])], axis=2)  # (b, 301, 2056)
        a = F.transpose(a, (0, 2, 1))[:, :, :, xp.newaxis]  # (b, 2056, 301, 1)

        x = F.concat((b, c))  # (b, 512, 301, 1)
        x = F.concat((x, a), axis=1)   # (b, 2568, 301, 1)
        x = F.transpose(x, (0, 2, 1, 3))[:, :, :, 0]  # (b, 301, 2568)

        y = self.fusion_stream(x)

        return y[:, 0, :, :, :], y[:, 1, :, :, :]

    def __init__(self, trained=None):

        super(Double_Complex_Net, self).__init__()

        with self.init_scope():

            # ===== Initialize variables ===== #
            self.speacker = 2
            self.audio_channels = 2

            if env.AUDIO_LEN % env.FC_ROW != 0:
                print("invalid fc layer parameter")
                import sys
                sys.exit(1)

            # ===== Audio Stream ===== #
            self.audio_stream = Audio_Stream(
                trained=None if trained is None else trained["audio_stream"])

            # ===== Visual Streams ===== #
            self.visual_stream = Visual_Stream(
                trained=None if trained is None else trained["visual_stream"])

            # ===== Fusion Stream ===== #
            self.fusion_stream = Fusion_Stream(
                trained=None if trained is None else trained["fusion_stream"]
            )
