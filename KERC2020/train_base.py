
import os
import torch
import numpy as np
import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from helper import load_args, seed_everything


class Trainer():
    def experiment_setup(self, args):
        raise NotImplementedError

    @staticmethod
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
        lr_logger = LearningRateLogger(logging_interval='epoch')
        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=logger,
                                                checkpoint_callback=checkpoint_callback,
                                                log_save_interval=args.log_save_interval,
                                                callbacks=[lr_logger])
        if args.is_training:
            trainer.fit(model, datamodule=dataloader)
        else:
            trainer.test(model, datamodule=dataloader)
