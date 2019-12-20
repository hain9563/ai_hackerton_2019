from torch import nn

from models.Flatten import Flatten


class CI(nn.Module):
    def __init__(self, features, channel):
        super(CI, self).__init__()
        self.features = features
        # self.se = SELayer(channel)
        self.cl = nn.Sequential(
                                nn.Conv2d(channel, 2048, 1, bias=False),
                                nn.AdaptiveAvgPool2d(1),
                                Flatten(),
                                nn.Linear(2048, 1000),
                                nn.BatchNorm1d(1000),
                                nn.ReLU(),
                                nn.Linear(1000, 1000),
                                nn.BatchNorm1d(1000),
                                nn.ReLU(),
                                nn.Linear(1000, 4)
                                )

    def forward(self, input):
        # se = self.se(input)
        return self.cl(input)

