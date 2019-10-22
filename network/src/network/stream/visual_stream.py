import chainer
import chainer.links as L
import chainer.functions as F

import env


class Visual_Stream(chainer.Chain):

    # @chainer.static_graph
    def __call__(self, visual):

        b = F.leaky_relu(self.bn1(self.conv1(visual)))
        b = F.leaky_relu(self.bn2(self.conv2(b)))
        b = F.leaky_relu(self.bn3(self.conv3(b)))
        b = F.leaky_relu(self.bn4(self.conv4(b)))
        b = F.leaky_relu(self.bn5(self.conv5(b)))
        b = F.leaky_relu(self.bn6(self.conv6(b)))  # (b, 256, 75, 1)
        b = F.resize_images(b, (env.AUDIO_LEN, 1))  # (b, 256, 301, 1)

        return b

    def __init__(self, trained=None):

        super(Visual_Stream, self).__init__()

        with self.init_scope():

            initial = chainer.initializers.HeNormal()

            self.conv1 = L.DilatedConvolution2D(
                in_channels=env.VIS_CHANNNEL, out_channels=256,
                stride=1, ksize=(7, 1), dilate=1, pad=(3, 0),
                nobias=True, initialW=initial)
            self.conv2 = L.DilatedConvolution2D(
                in_channels=256, out_channels=256,
                stride=1, ksize=(5, 1), dilate=1, pad=(2, 0),
                nobias=True, initialW=initial)
            self.conv3 = L.DilatedConvolution2D(
                in_channels=256, out_channels=256,
                stride=1, ksize=(5, 1), dilate=(2, 1), pad=(4, 0),
                nobias=True, initialW=initial)
            self.conv4 = L.DilatedConvolution2D(
                in_channels=256, out_channels=256,
                stride=1, ksize=(5, 1), dilate=(4, 1), pad=(8, 0),
                nobias=True, initialW=initial)
            self.conv5 = L.DilatedConvolution2D(
                in_channels=256, out_channels=256,
                stride=1, ksize=(5, 1), dilate=(8, 1), pad=(16, 0),
                nobias=True, initialW=initial)
            self.conv6 = L.DilatedConvolution2D(
                in_channels=256, out_channels=256,
                stride=1, ksize=(5, 1), dilate=(16, 1), pad=(32, 0),
                nobias=True, initialW=initial)

            self.bn1 = L.BatchNormalization(256)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(256)
            self.bn4 = L.BatchNormalization(256)
            self.bn5 = L.BatchNormalization(256)
            self.bn6 = L.BatchNormalization(256)

            if trained is not None:

                self.conv1.W = trained["conv1"].W
                self.conv2.W = trained["conv2"].W
                self.conv3.W = trained["conv3"].W
                self.conv4.W = trained["conv4"].W
                self.conv5.W = trained["conv5"].W
                self.conv6.W = trained["conv6"].W

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
