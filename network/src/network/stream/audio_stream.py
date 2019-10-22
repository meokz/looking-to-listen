import chainer
import chainer.links as L
import chainer.functions as F

import env


class Audio_Stream(chainer.Chain):

    # @chainer.static_graph
    def __call__(self, spec):

        a = F.leaky_relu(self.bn1(self.conv1(spec)))
        a = F.leaky_relu(self.bn2(self.conv2(a)))
        a = F.leaky_relu(self.bn3(self.conv3(a)))
        a = F.leaky_relu(self.bn4(self.conv4(a)))
        a = F.leaky_relu(self.bn5(self.conv5(a)))
        a = F.leaky_relu(self.bn6(self.conv6(a)))
        a = F.leaky_relu(self.bn7(self.conv7(a)))
        a = F.leaky_relu(self.bn8(self.conv8(a)))
        a = F.leaky_relu(self.bn9(self.conv9(a)))
        a = F.leaky_relu(self.bn10(self.conv10(a)))
        a = F.leaky_relu(self.bn11(self.conv11(a)))
        a = F.leaky_relu(self.bn12(self.conv12(a)))
        a = F.leaky_relu(self.bn13(self.conv13(a)))
        a = F.leaky_relu(self.bn14(self.conv14(a)))
        a = F.leaky_relu(self.bn15(self.conv15(a)))

        return a  # (b, 8, 301, 257)

    def __init__(self, trained=None):

        super(Audio_Stream, self).__init__()

        with self.init_scope():

            initial = chainer.initializers.HeNormal()

            self.conv1 = L.DilatedConvolution2D(
                in_channels=env.AUDIO_CHANNELS, out_channels=96,
                stride=1, ksize=(1, 7), dilate=(1, 1), pad=(0, 3),
                nobias=True, initialW=initial)
            self.conv2 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(7, 1), dilate=(1, 1), pad=(3, 0),
                nobias=True, initialW=initial)
            self.conv3 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(1, 1), pad=(2, 2),
                nobias=True, initialW=initial)
            self.conv4 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(2, 1), pad=(4, 2),
                nobias=True, initialW=initial)
            self.conv5 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(4, 1), pad=(8, 2),
                nobias=True, initialW=initial)
            self.conv6 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(8, 1), pad=(16, 2),
                nobias=True, initialW=initial)
            self.conv7 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(16, 1), pad=(32, 2),
                nobias=True, initialW=initial)
            self.conv8 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(32, 1), pad=(64, 2),
                nobias=True, initialW=initial)
            self.conv9 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(1, 1), pad=(2, 2),
                nobias=True, initialW=initial)
            self.conv10 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(2, 2), pad=(4, 4),
                nobias=True, initialW=initial)
            self.conv11 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(4, 4), pad=(8, 8),
                nobias=True, initialW=initial)
            self.conv12 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(8, 8), pad=(16, 16),
                nobias=True, initialW=initial)
            self.conv13 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(16, 16), pad=(32, 32),
                nobias=True, initialW=initial)
            self.conv14 = L.DilatedConvolution2D(
                in_channels=96, out_channels=96,
                stride=1, ksize=(5, 5), dilate=(32, 32), pad=(64, 64),
                nobias=True, initialW=initial)
            self.conv15 = L.DilatedConvolution2D(
                in_channels=96, out_channels=8,
                stride=1, ksize=(1, 1), dilate=(1, 1), pad=(0, 0),
                nobias=True, initialW=initial)

            self.bn1 = L.BatchNormalization(96)
            self.bn2 = L.BatchNormalization(96)
            self.bn3 = L.BatchNormalization(96)
            self.bn4 = L.BatchNormalization(96)
            self.bn5 = L.BatchNormalization(96)
            self.bn6 = L.BatchNormalization(96)
            self.bn7 = L.BatchNormalization(96)
            self.bn8 = L.BatchNormalization(96)
            self.bn9 = L.BatchNormalization(96)
            self.bn10 = L.BatchNormalization(96)
            self.bn11 = L.BatchNormalization(96)
            self.bn12 = L.BatchNormalization(96)
            self.bn13 = L.BatchNormalization(96)
            self.bn14 = L.BatchNormalization(96)
            self.bn15 = L.BatchNormalization(8)

            if trained is not None:

                self.conv1.W = trained["conv1"].W
                self.conv2.W = trained["conv2"].W
                self.conv3.W = trained["conv3"].W
                self.conv4.W = trained["conv4"].W
                self.conv5.W = trained["conv5"].W
                self.conv6.W = trained["conv6"].W
                self.conv7.W = trained["conv7"].W
                self.conv8.W = trained["conv8"].W
                self.conv9.W = trained["conv9"].W
                self.conv10.W = trained["conv10"].W
                self.conv11.W = trained["conv11"].W
                self.conv12.W = trained["conv12"].W
                self.conv13.W = trained["conv13"].W
                self.conv14.W = trained["conv14"].W
                self.conv15.W = trained["conv15"].W

                self.bn1.gamma = trained["bn1"].gamma
                self.bn1.beta = trained["bn1"].beta
                self.bn1.avg_mean = trained["bn1"].avg_mean
                self.bn1.avg_var = trained["bn1"].avg_var

                self.bn2.gamma = trained["bn2"].gamma
                self.bn2.beta = trained["bn2"].beta
                self.bn2.avg_mean = trained["bn2"].avg_mean
                self.bn2.avg_var = trained["bn2"].avg_var

                self.bn3.gamma = trained["bn3"].gamma
                self.bn3.beta = trained["bn3"].beta
                self.bn3.avg_mean = trained["bn3"].avg_mean
                self.bn3.avg_var = trained["bn3"].avg_var

                self.bn4.gamma = trained["bn4"].gamma
                self.bn4.beta = trained["bn4"].beta
                self.bn4.avg_mean = trained["bn4"].avg_mean
                self.bn4.avg_var = trained["bn4"].avg_var

                self.bn5.gamma = trained["bn5"].gamma
                self.bn5.beta = trained["bn5"].beta
                self.bn5.avg_mean = trained["bn5"].avg_mean
                self.bn5.avg_var = trained["bn5"].avg_var

                self.bn6.gamma = trained["bn6"].gamma
                self.bn6.beta = trained["bn6"].beta
                self.bn6.avg_mean = trained["bn6"].avg_mean
                self.bn6.avg_var = trained["bn6"].avg_var

                self.bn7.gamma = trained["bn7"].gamma
                self.bn7.beta = trained["bn7"].beta
                self.bn7.avg_mean = trained["bn7"].avg_mean
                self.bn7.avg_var = trained["bn7"].avg_var

                self.bn8.gamma = trained["bn8"].gamma
                self.bn8.beta = trained["bn8"].beta
                self.bn8.avg_mean = trained["bn8"].avg_mean
                self.bn8.avg_var = trained["bn8"].avg_var

                self.bn9.gamma = trained["bn9"].gamma
                self.bn9.beta = trained["bn9"].beta
                self.bn9.avg_mean = trained["bn9"].avg_mean
                self.bn9.avg_var = trained["bn9"].avg_var

                self.bn10.gamma = trained["bn10"].gamma
                self.bn10.beta = trained["bn10"].beta
                self.bn10.avg_mean = trained["bn10"].avg_mean
                self.bn10.avg_var = trained["bn10"].avg_var

                self.bn11.gamma = trained["bn11"].gamma
                self.bn11.beta = trained["bn11"].beta
                self.bn11.avg_mean = trained["bn11"].avg_mean
                self.bn11.avg_var = trained["bn11"].avg_var

                self.bn12.gamma = trained["bn12"].gamma
                self.bn12.beta = trained["bn12"].beta
                self.bn12.avg_mean = trained["bn12"].avg_mean
                self.bn12.avg_var = trained["bn12"].avg_var

                self.bn13.gamma = trained["bn13"].gamma
                self.bn13.beta = trained["bn13"].beta
                self.bn13.avg_mean = trained["bn13"].avg_mean
                self.bn13.avg_var = trained["bn13"].avg_var

                self.bn14.gamma = trained["bn14"].gamma
                self.bn14.beta = trained["bn14"].beta
                self.bn14.avg_mean = trained["bn14"].avg_mean
                self.bn14.avg_var = trained["bn14"].avg_var

                self.bn15.gamma = trained["bn15"].gamma
                self.bn15.beta = trained["bn15"].beta
                self.bn15.avg_mean = trained["bn15"].avg_mean
                self.bn15.avg_var = trained["bn15"].avg_var
