
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
        # return torch.optim.SGD(self.parameters(),
        #                        lr=0.001,
        #                        momentum=0.9)

    # def configure_optimizers(self):
    #     opt = torch.optim.Adam(self.parameters(),
    #                            lr=self.learning_rate)
    #     lr_sched = {'scheduler': BoundingExponentialLR(opt, 0.5),
    #                 'interval': 'epoch',  # or 'epoch'
    #                 'monitor': 'val_loss',
    #                 'frequency': 10,  # call scheduler every x steps
    #                 }
    #     return [opt], [lr_sched]

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
