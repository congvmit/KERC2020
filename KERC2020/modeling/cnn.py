# from torch.nn import Conv2d, Dropout, Flatten, MaxPool2d, Linear
# import torch
# import pytorch_lightning as pl
# from pytorch_lightning.metrics.functional import accuracy
# from backbone.mobilenet import MobileNetV2Backbone

import torch
import torch.nn as nn
from lr_schedulers import BoundingExponentialLR
from modeling.base import BaseModel
from modeling.backbone import resnet


class StaticModel(BaseModel):

    def __init__(self, backbone,
                 learning_rate=1e-2,
                 dropout_rate=0.2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(2048)
        self.linear = nn.Linear(2048, 256)
        self.arousal_ln = nn.Linear(256, 1)
        self.stress_ln = nn.Linear(256, 1)
        self.valence_ln = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        out = self.backbone(x)
        x = torch.flatten(out, start_dim=1)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.bn(x)

        x = self.linear(x)
        x = self.relu(x)

        varousal = self.arousal_ln(x)
        stress = self.stress_ln(x)
        valence = self.valence_ln(x)

        return varousal, stress, valence

    def get_loss(self):
        return nn.MSELoss()

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(),
                              lr=self.learning_rate)
        lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5),
                    'interval': 'epoch',  # or 'epoch'
                    'monitor': 'val_loss',
                    'frequency': 1,  # call scheduler every x steps
                    }
        return [opt], [lr_sched]


def load_resnet50(*args, **kwargs):
    backbone = resnet.resnet50(pretrained=True,
                               include_top=False)
    return StaticModel(backbone=backbone, *args, **kwargs)
