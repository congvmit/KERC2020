
from torch.nn import Conv2d, Dropout, Flatten, MaxPool2d, Linear
import torch
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from backbone.mobilenet import MobileNetV2Backbone
from backbone.resnet import resnet50


class BoundingExponentialLR(ExponentialLR):
    """Decays the learning rate of each parameter group by gamma every epoch.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, gamma, min_lr=0.001, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, gamma, last_epoch)

    def _compute_lr(self, base_lr):
        _base_lr = base_lr * self.gamma ** self.last_epoch
        if _base_lr <= self.min_lr:
            return self.min_lr
        else:
            return _base_lr

    def _get_closed_form_lr(self):
        return [self._compute_lr(base_lr)
                for base_lr in self.base_lrs]


class EMORegressor(pl.LightningModule):
    def __init__(self, arch='resnet50',
                 learning_rate=1e-3,
                 batch_size=32,
                 freeze_backbone=False, **kwargs):

        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if arch == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            last_dim = 2048
        elif arch == 'mobilenetv2':
            self.backbone = MobileNetV2Backbone()
            last_dim = 1280
        else:
            raise ValueError('Backbone must be one of [resnet50, mobilenetv2]')

        self.arousal_ln = Linear(last_dim, 1)
        self.stress_ln = Linear(last_dim, 1)
        self.valence_ln = Linear(last_dim, 1)

        self.example_input_array = torch.zeros([1, 3, 256, 256])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        varousal = self.arousal_ln(x)
        stress = self.stress_ln(x)
        valence = self.valence_ln(x)
        return varousal, stress, valence

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(),
    #                             lr=self.hparams.learning_rate)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(),
                               lr=self.learning_rate)
        lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5),
                    'interval': 'epoch',  # or 'epoch'
                    'monitor': 'val_loss',
                    'frequency': 10,  # call scheduler every x steps
                    }
        return [opt], [lr_sched]

    def training_step(self, batch, batch_idx):
        data = batch
        x = data['img']
        arousal = data['arousal']
        stress = data['stress']
        valence = data['valence']

        pred_arousal, pred_stress, pred_valence = self.forward(x)

        arousal_loss = torch.nn.functional.mse_loss(pred_arousal, arousal)
        stress_loss = torch.nn.functional.mse_loss(pred_stress, stress)
        valence_loss = torch.nn.functional.mse_loss(pred_valence, valence)

        loss = (arousal_loss + 2 * stress_loss + valence_loss) / 4
        result = pl.TrainResult(minimize=loss)
        result.log_dict({
            'arousal_loss': arousal_loss,
            'stress_loss': stress_loss,
            'valence_loss': valence_loss,
            'loss': loss
        }, logger=False)

        return result

    def training_epoch_end(self, output_results):
        arousal_loss = torch.mean(output_results['arousal_loss'])
        stress_loss = torch.mean(output_results['stress_loss'])
        valence_loss = torch.mean(output_results['valence_loss'])
        loss = torch.mean(output_results['loss'])

        self.logger.experiment.add_scalars("Losses", {
            'train_arousal': arousal_loss,
            'train_stress': stress_loss,
            'train_valence': valence_loss,
            'train_total': loss
        }, global_step=self.current_epoch)

        result = pl.TrainResult()
        return result

    def validation_step(self, batch, batch_idx):
        data = batch
        x = data['img']
        arousal = data['arousal']
        stress = data['stress']
        valence = data['valence']

        pred_arousal, pred_stress, pred_valence = self.forward(x)
        arousal_loss = torch.nn.functional.mse_loss(pred_arousal, arousal)
        stress_loss = torch.nn.functional.mse_loss(pred_stress, stress)
        valence_loss = torch.nn.functional.mse_loss(pred_valence, valence)
        loss = (arousal_loss + 2 * stress_loss + valence_loss) / 4

        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({
            'val_arousal': arousal_loss,
            'val_stress': stress_loss,
            'val_valence': valence_loss,
            'val_loss': loss
        }, logger=False)
        return result

    def validation_epoch_end(self, output_results):
        arousal_loss = torch.mean(output_results['val_arousal'])
        stress_loss = torch.mean(output_results['val_stress'])
        valence_loss = torch.mean(output_results['val_valence'])
        loss = torch.mean(output_results['val_loss'])

        self.logger.experiment.add_scalars("Losses", {
            'val_arousal': arousal_loss,
            'val_stress': stress_loss,
            'val_valence': valence_loss,
            'val_loss': loss
        }, global_step=self.current_epoch)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, logger=False, prog_bar=True)
        return result


if __name__ == "__main__":
    from torchsummary import summary
    model = EMORegressor()
    print(summary(model, input_size=(3, 256, 256), device='cpu'))
    inp = torch.ones([1, 3, 256, 256])
    y = model(inp)
