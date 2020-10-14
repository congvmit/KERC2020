from .backbone.mobilenet import MobileNetV2Backbone
import torch
import torch.nn as nn
from lr_schedulers import BoundingExponentialLR
from modeling.base import BaseClassifier
from modeling.base import BaseMultiTaskClassifier
from modeling.backbone import resnet


# =================================================================================
class MultiTaskClassifier(BaseMultiTaskClassifier):

    def __init__(self, backbone,
                 learning_rate=1e-2,
                 dropout_rate=0.2,
                 n_classes=7,
                 backbone_flatten_dim=2048,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.backbone = backbone
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(backbone_flatten_dim)

        self.emotion_ln = nn.Linear(backbone_flatten_dim, n_classes)
        self.landmark_ln = nn.Linear(backbone_flatten_dim, 68 * 2)
        self.valence_ln = nn.Linear(backbone_flatten_dim, 1)
        self.arousal_ln = nn.Linear(backbone_flatten_dim, 1)

    def forward(self, x):
        out = self.backbone(x)
        x = self.flatten(out)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)

        emotion_pred = self.emotion_ln(x)
        landmarks_pred = self.landmark_ln(x)
        valence_pred = self.valence_ln(x)
        arousal_pred = self.arousal_ln(x)
        return emotion_pred, valence_pred, arousal_pred, landmarks_pred

    def get_loss(self):
        return {
            'classification_loss': nn.CrossEntropyLoss(),
            'arousal_loss': nn.MSELoss(),
            'valence_loss': nn.MSELoss(),
            'landmarks': nn.MSELoss()
        }

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(),
                              lr=self.learning_rate)
        lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5, initial_lr=self.learning_rate),
                    'interval': 'epoch',  # or 'epoch'
                    'monitor': 'val_loss',
                    'frequency': 1,  # call scheduler every x steps
                    }
        return [opt], [lr_sched]


def load_affectnet_resnet50(*args, **kwargs):
    backbone = resnet.resnet50(pretrained=True,
                               include_top=False)
    return MultiTaskClassifier(backbone=backbone, *args, **kwargs)


# =================================================================================
class Classifier(BaseClassifier):

    def __init__(self, backbone,
                 learning_rate=1e-2,
                 dropout_rate=0.2,
                 n_classes=7,
                 backbone_flatten_dim=2048,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        # self.save_hyperparameters()
        self.backbone = backbone
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(backbone_flatten_dim)
        self.linear = nn.Linear(backbone_flatten_dim, n_classes)

    def forward(self, x):
        out = self.backbone(x)
        x = self.flatten(out)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear(x)
        return x

    def get_loss(self):
        return nn.CrossEntropyLoss()

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.parameters(),
                              lr=self.learning_rate)
        lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5, initial_lr=self.learning_rate),
                    'interval': 'epoch',  # or 'epoch'
                    'monitor': 'val_loss',
                    'frequency': 1,  # call scheduler every x steps
                    }
        return [opt], [lr_sched]


def load_kerc2019_resnet50(*args, **kwargs):
    backbone = resnet.resnet50(pretrained=True,
                               include_top=False)
    return Classifier(backbone=backbone, *args, **kwargs)


def load_kerc2019_mobilenetv2(*args, **kwargs):
    backbone = MobileNetV2Backbone()
    return Classifier(backbone=backbone, backbone_flatten_dim=1280, *args, **kwargs)
