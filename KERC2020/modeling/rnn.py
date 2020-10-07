from .base import BaseModel
import torch.nn as nn
from .backbone import ConvLSTM
import torch
from lr_schedulers import BoundingExponentialLR


class VisionRNNModel(BaseModel):

    def __init__(self, rnn_model, *args, **kwargs):
        super().__init__()
        self.example_input_array = torch.zeros([1, 68, 3, 96, 96])
        #  B, T, C, H, W or T, B, C, H, W
        self.rnn_model = rnn_model
        self.arousal_ln = nn.Linear(576, 1)
        self.stress_ln = nn.Linear(576, 1)
        self.valence_ln = nn.Linear(576, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out, last_states = self.rnn_model(x)
        x = last_states[0][1].view(-1, 576)
        x = self.relu(x)

        varousal = self.arousal_ln(x)
        stress = self.stress_ln(x)
        valence = self.valence_ln(x)
        return varousal, stress, valence

    def get_loss(self):
        return nn.L1Loss()

    # def configure_optimizers(self):
    #     opt = torch.optim.SGD(self.parameters(),
    #                           lr=0.01, momentum=0.9, nesterov=True)
    #     lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5),
    #                 'interval': 'epoch',  # or 'epoch'
    #                 'monitor': 'val_loss',
    #                 'frequency': 10,  # call scheduler every x steps
    #                 }
    #     return [opt], [lr_sched]


def load_convlstm():
    convlstm = ConvLSTM(input_dim=3,
                        hidden_dim=16,
                        kernel_size=(3, 3),
                        num_layers=1,
                        batch_first=True)
    return VisionRNNModel(rnn_model=convlstm)
