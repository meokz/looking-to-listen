import chainer
import chainer.functions as F

from network.stream.audio_stream import Audio_Stream
from network.stream.fusion_stream import Fusion_Stream
from network.loss import evaluate_loss

import modules.operation as op
from env import xp


class Audio_Only_Net(chainer.Chain):

    def __call__(self, noise, clean):

        noise = xp.asarray(noise).astype(xp.float32)
        clean = xp.asarray(clean).astype(xp.float32)

        compressed_noise, _ = op.compress_audio(noise)
        compressed_clean, _ = op.compress_audio(clean)

        mask, _ = self.estimate_mask(spec=compressed_noise)

        separated = op.mul(mask, compressed_noise)

        loss = evaluate_loss(self, separated, compressed_clean)

        return loss

    def estimate_mask(self, spec):

        # ===== Audio Stream ===== #
        a = self.audio_stream(spec)
        a = F.concat([a[:, i, :, :] for i in range(a.shape[1])], axis=2)  # (b, 301, 2056)
        x = a  # (b, 301, 2568)

        y = self.fusion_stream(x)

        # y[batch_size, env.OUTPUT_MASK,  env.AUDIO_CHANNELS, env.AUDIO_LEN, 257]
        return y[:, 0, :, :, :], 1 - y[:, 0, :, :, :]

    def __init__(self, trained=None):

        super(Audio_Only_Net, self).__init__()

        with self.init_scope():

            # ===== Audio Stream ===== #
            self.audio_stream = Audio_Stream(
                trained=None if trained is None else trained["audio_stream"])

            self.fusion_stream = Fusion_Stream(
                trained=None if trained is None else trained["fusion_stream"]
            )
