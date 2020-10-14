
import os
import torch
import numpy as np
import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from helper import load_args, seed_everything
from torch.utils.data import DataLoader, Dataset


class BaseDataLoader(pl.LightningDataModule):

    def __init__(self,
                 train_dataset: Dataset = None,
                 val_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 batch_size: int = 2,
                 num_workers: int = 4,
                 gpus: int = 0,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device_config = {}
        if gpus != 0:
            self.device_config.update({
                'pin_memory': True
            })

    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('train_dataset must be specified')
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **self.device_config)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise ValueError('val_dataset must be specified')
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          drop_last=False,
                          **self.device_config)

    def test_dataloader(self):
        if self.test_dataset is None:
            raise ValueError('test_dataset must be specified')
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          **self.device_config)


class Trainer():
    def experiment_setup(self, args):
        ''' Receive args then initialize dataloader and model

            Parameters:
                args

            Return:
                args, dataloader, model
        '''
        raise NotImplementedError

    def start(self):
        '''Template to train and evaluate'''
        args = load_args()
        seed_everything(args)
        args, dataloader, model = self.experiment_setup(args)
        # Training
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            save_top_k=3,
            verbose=True,
            monitor='val_loss',
            mode='min',
            prefix=''
        )

        logger = TensorBoardLogger(save_dir=args.tensorboard_dir,
                                   name=args.ckpt_prefix)
        lr_logger = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=logger,
                                                checkpoint_callback=checkpoint_callback,
                                                callbacks=[lr_logger])

        if args.is_training:
            print("Starting training...")
            trainer.fit(model, datamodule=dataloader)
        else:
            trainer.test(model, datamodule=dataloader)
