
import numpy as np
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
import torch.nn as nn
import torch


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        data = batch
        x = data['image']
        arousal = data['arousal']
        stress = data['stress']
        valence = data['valence']

        pred_arousal, pred_stress, pred_valence = self.forward(x)

        arousal_loss = self.get_loss()(pred_arousal, arousal)
        stress_loss = self.get_loss()(pred_stress, stress)
        valence_loss = self.get_loss()(pred_valence, valence)

        loss = (arousal_loss + stress_loss + valence_loss) / 3.
        result = pl.TrainResult(minimize=loss)
        result.log_dict({
            'a_loss': arousal_loss,
            's_loss': stress_loss,
            'v_loss': valence_loss,
            'loss': loss
        }, logger=False, prog_bar=True)  # , on_step=True, on_epoch=False)
        return result

    def training_epoch_end(self, output_results):
        arousal_loss = torch.mean(output_results['a_loss'])
        stress_loss = torch.mean(output_results['s_loss'])
        valence_loss = torch.mean(output_results['v_loss'])
        loss = torch.mean(output_results['loss'])

        # https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
        self.logger.experiment.add_scalars("Losses", {
            'train_arousal': arousal_loss,
            'train_stress': stress_loss,
            'train_valence': valence_loss,
            'train_total': loss
        }, global_step=self.current_epoch)

        result = pl.TrainResult()
        result.log_dict({
            'a_loss': arousal_loss,
            's_loss': stress_loss,
            'v_loss': valence_loss,
        }, logger=False, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        arousal = batch['arousal']
        stress = batch['stress']
        valence = batch['valence']
        # face_files = batch['face_file']

        pred_arousal, pred_stress, pred_valence = self.forward(x)
        arousal_loss = self.get_loss()(pred_arousal, arousal)
        stress_loss = self.get_loss()(pred_stress, stress)
        valence_loss = self.get_loss()(pred_valence, valence)
        loss = (arousal_loss + stress_loss + valence_loss) / 3.

        result = pl.EvalResult()
        result.log_dict({
            'val_a': arousal_loss,
            'val_s': stress_loss,
            'val_v': valence_loss,
            'val_loss': loss,
        }, logger=False)
        return result

    def validation_epoch_end(self, output_results):
        arousal_loss = torch.mean(output_results['val_a'])
        stress_loss = torch.mean(output_results['val_s'])
        valence_loss = torch.mean(output_results['val_v'])
        loss = torch.mean(output_results['val_loss'])

        self.logger.experiment.add_scalars("Losses", {
            'val_arousal': arousal_loss,
            'val_stress': stress_loss,
            'val_valence': valence_loss,
            'val_loss': loss
        }, global_step=self.current_epoch)

        result = pl.EvalResult(checkpoint_on=loss)
        # result.log('val_loss', loss, logger=False, prog_bar=True)
        result.log_dict({
            'val_a': arousal_loss,
            'val_s': stress_loss,
            'val_v': valence_loss,
            'val_loss': loss
        }, logger=False, prog_bar=True)

        return result

# =================================================================================


class BaseMultiTaskClassifier(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = batch['image']
        emotion_true = batch['expression']

        valence_true = batch['valence']
        arousal_true = batch['arousal']
        landmarks_true = batch['landmarks']

        emotion_pred, valence_pred, arousal_pred, landmarks_pred = self.forward(x)
        # Loss

        emotion_loss = self.get_loss()['classification_loss'](emotion_pred, emotion_true)
        arousal_loss = self.get_loss()['arousal_loss'](arousal_pred, arousal_true)
        valence_loss = self.get_loss()['valence_loss'](valence_pred, valence_true)
        landmarks_loss = self.get_loss()['landmarks'](landmarks_pred, landmarks_true)

        loss = (emotion_loss + arousal_loss + valence_loss + landmarks_loss) / 4

        # Accuracy
        emotion_pred = torch.argmax(emotion_pred, dim=1).flatten()
        acc = (emotion_true == emotion_pred).sum().float() / emotion_true.shape[0]

        train_log = {
            'acc': acc,
            'loss': loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss,
            'landmarks_loss': landmarks_loss
        }

        self.log('acc', acc, logger=False, prog_bar=True)
        return train_log

    def training_epoch_end(self, output_results):
        acc = torch.mean(torch.stack([result['acc'] for result in output_results]))
        loss = torch.mean(torch.stack([result['loss'] for result in output_results]))

        emotion_loss = torch.mean(torch.stack([result['emotion_loss'] for result in output_results]))
        arousal_loss = torch.mean(torch.stack([result['arousal_loss'] for result in output_results]))
        valence_loss = torch.mean(torch.stack([result['valence_loss'] for result in output_results]))
        landmarks_loss = torch.mean(torch.stack([result['landmarks_loss'] for result in output_results]))

        # https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
        self.logger.experiment.add_scalars("Accuracy", {
            'train_acc': acc,
        }, global_step=self.current_epoch)

        self.logger.experiment.add_scalars("Losses", {
            'train_loss': loss,
            'train_emotion_loss': emotion_loss,
            'train_arousal_loss': arousal_loss,
            'train_valence_loss': valence_loss,
            'train_landmarks_loss': landmarks_loss
        }, global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        emotion_true = batch['expression']

        valence_true = batch['valence']
        arousal_true = batch['arousal']
        landmarks_true = batch['landmarks']

        emotion_pred, valence_pred, arousal_pred, landmarks_pred = self.forward(x)

        # Loss
        emotion_loss = self.get_loss()['classification_loss'](emotion_pred, emotion_true)
        arousal_loss = self.get_loss()['arousal_loss'](arousal_pred, arousal_true)
        valence_loss = self.get_loss()['valence_loss'](valence_pred, valence_true)
        landmarks_loss = self.get_loss()['landmarks'](landmarks_pred, landmarks_true)

        loss = (emotion_loss + arousal_loss + valence_loss + landmarks_loss) / 4

        # Accuracy
        emotion_pred = torch.argmax(emotion_pred, dim=1).flatten()
        acc = (emotion_true == emotion_pred).sum().float() / emotion_true.shape[0]

        val_log = {
            'val_acc': acc,
            'val_loss': loss,
            'emotion_loss': emotion_loss,
            'arousal_loss': arousal_loss,
            'valence_loss': valence_loss,
            'landmarks_loss': landmarks_loss
        }

        self.log('val_acc', acc, logger=False, prog_bar=True)
        self.log('val_loss', loss, logger=False, prog_bar=True)
        return val_log

    def validation_epoch_end(self, output_results):
        acc = torch.mean(torch.stack([result['val_acc'] for result in output_results]))
        loss = torch.mean(torch.stack([result['val_loss'] for result in output_results]))

        emotion_loss = torch.mean(torch.stack([result['emotion_loss'] for result in output_results]))
        arousal_loss = torch.mean(torch.stack([result['arousal_loss'] for result in output_results]))
        valence_loss = torch.mean(torch.stack([result['valence_loss'] for result in output_results]))
        landmarks_loss = torch.mean(torch.stack([result['landmarks_loss'] for result in output_results]))

        self.logger.experiment.add_scalars("Losses", {
            'val_loss': loss,
            'val_emotion_loss': emotion_loss,
            'val_arousal_loss': arousal_loss,
            'val_valence_loss': valence_loss,
            'val_landmarks_loss': landmarks_loss
        }, global_step=self.current_epoch)

        self.logger.experiment.add_scalars("Accuracy", {
            'val_acc': acc
        }, global_step=self.current_epoch)


# =================================================================================
class BaseClassifier(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y_true = batch['label']
        y_pred = self.forward(x)
        # Loss
        loss = self.get_loss()(y_pred, y_true)
        # Accuracy
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        acc = (y_true == y_pred).sum().float() / y_true.shape[0]

        train_log = {'acc': acc,
                     'loss': loss}
        self.log('acc', acc, logger=False, prog_bar=True)
        return train_log

    def training_epoch_end(self, output_results):
        acc = torch.mean(torch.stack([result['acc'] for result in output_results]))
        loss = torch.mean(torch.stack([result['loss'] for result in output_results]))

        # https://www.learnopencv.com/tensorboard-with-pytorch-lightning/
        self.logger.experiment.add_scalars("Accuracy", {
            'train_acc': acc,
        }, global_step=self.current_epoch)

        self.logger.experiment.add_scalars("Losses", {
            'train_loss': loss
        }, global_step=self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y_true = batch['label']
        y_pred = self.forward(x)

        # Loss
        loss = self.get_loss()(y_pred, y_true)

        # Accuracy
        y_pred = torch.argmax(y_pred, dim=1).flatten()
        acc = (y_true == y_pred).sum().float() / y_true.shape[0]

        val_log = {
            'val_acc': acc,
            'val_loss': loss
        }
        self.log_dict(val_log, logger=False, prog_bar=True)
        return val_log

    def validation_epoch_end(self, output_results):
        acc = torch.mean(torch.stack([result['val_acc'] for result in output_results]))
        loss = torch.mean(torch.stack([result['val_loss'] for result in output_results]))

        self.logger.experiment.add_scalars("Losses", {
            'val_loss': loss
        }, global_step=self.current_epoch)

        self.logger.experiment.add_scalars("Accuracy", {
            'val_acc': acc
        }, global_step=self.current_epoch)
