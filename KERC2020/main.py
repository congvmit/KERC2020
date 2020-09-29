
import os
import torch
import numpy as np
import pytorch_lightning as pl
from modeling import EMORegressor
from dataloader import EMODataModule
from argparse import ArgumentParser

import pprint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def load_args():
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = pl.LightningDataModule.add_argparse_args(parser)
    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--weight_path', default='', type=str)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--cuda_benchmark', action='store_true')
    args = parser.parse_args()
    print('-' * 100)
    pprint.pprint(vars(args))
    print('-' * 100)
    return args


def cli_main():
    args = load_args()
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.cuda_benchmark

    # Dataloader
    data_loader = EMODataModule(data_dir=args.data_dir,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # Model
    model = EMORegressor(**vars(args))

    if args.weight_path:
        print('Loading weights from {}'.format(args.weight_path))
        state_dict = torch.load(args.weight_path)
        model.load_state_dict(state_dict, strict=False)

    # Training
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    lr_logger = LearningRateLogger(logging_interval='epoch')

    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.max_epochs,
                         limit_train_batches=args.limit_train_batches,
                         default_root_dir=args.default_root_dir,
                         checkpoint_callback=checkpoint_callback,
                         callbacks=[lr_logger]
                         )
    if args.is_training:
        trainer.fit(model, datamodule=data_loader)
    else:
        trainer.test(model, datamodule=data_loader)


if __name__ == '__main__':
    cli_main()
