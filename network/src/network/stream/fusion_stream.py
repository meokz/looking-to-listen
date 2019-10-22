import chainer
import chainer.links as L
import chainer.functions as F

import env


class Fusion_Stream(chainer.Chain):

    def __call__(self, x):

        # x = (b, 301, 2568)
        batch_size = x.shape[0]

        # Array to List
        xs = [i for i in x]
        ys = self.lstm(hx=None, cx=None, xs=xs)[2]
        # List to Array
        y = F.stack(ys)  # (b, 301, 2*2*257)
        y = F.leaky_relu(self.bn0(y))

        y = F.reshape(
            y, shape=(batch_size * int(env.AUDIO_LEN / env.FC_ROW), -1))

        y = F.leaky_relu(self.bn1(self.fc1(y)))
        y = F.leaky_relu(self.bn2(self.fc2(y)))
        y = F.sigmoid(self.bn3(self.fc3(y)))

        y = F.reshape(
            y, shape=(batch_size, env.OUTPUT_MASK,  env.AUDIO_CHANNELS, env.AUDIO_LEN, 257))

        return y

    def __init__(self, trained=None):

        super(Fusion_Stream, self).__init__()

        with self.init_scope():

            initial = chainer.initializers.HeNormal()

            if env.AUDIO_LEN % env.FC_ROW != 0:
                print("invalid fc layer parameter")
                import sys
                sys.exit(1)

            self.lstm = L.NStepBiLSTM(
                n_layers=1, in_size=2056+env.INPUT_FACE*256, out_size=300, dropout=0.0)

            self.fc1 = L.Linear(
                in_size=600*env.FC_ROW, out_size=600*env.FC_ROW,
                nobias=True, initialW=initial)
            self.fc2 = L.Linear(
                in_size=600*env.FC_ROW, out_size=600*env.FC_ROW,
                nobias=True, initialW=initial)
            self.fc3 = L.Linear(
                in_size=600*env.FC_ROW,
                out_size=env.OUTPUT_MASK*env.AUDIO_CHANNELS*257*env.FC_ROW,
                nobias=True, initialW=initial)

            self.bn0 = L.BatchNormalization(301)
            self.bn1 = L.BatchNormalization(600*env.FC_ROW)
            self.bn2 = L.BatchNormalization(600*env.FC_ROW)
            self.bn3 = L.BatchNormalization(
                env.OUTPUT_MASK*env.AUDIO_CHANNELS*257*env.FC_ROW)

            if trained is not None:
                self.lstm = trained["lstm"]
                # self.lstm.ws = trained["lstm"].ws
                # self.lstm.bs = trained["lstm"].bs

                self.fc1.W = trained["fc1"].W
                self.fc2.W = trained["fc2"].W
                self.fc3.W = trained["fc3"].W

                self.bn0.gamma = trained["bn0"].gamma
                self.bn0.beta = trained["bn0"].beta
                self.bn0.avg_mean = trained["bn0"].avg_mean
                self.bn0.avg_var = trained["bn0"].avg_var

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
