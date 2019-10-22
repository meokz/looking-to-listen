import chainer
import chainer.functions as F


def evaluate_loss(model, compressed_separated, compressed_clean):

    loss = F.mean_squared_error(compressed_separated, compressed_clean)
    chainer.reporter.report({"loss": loss.data}, model)

    return loss
