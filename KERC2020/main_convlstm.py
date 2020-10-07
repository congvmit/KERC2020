

import os
import torch
import numpy as np
import pprint

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from helper import load_args, seed_everything


def cli_main():
    args = load_args()
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.cuda_benchmark

    train_data = KERCVideoDataset(data_dir='/home/congvm/Dataset/',
                                  csv_path='/home/congvm/Dataset/dataset/train_faces.csv',
                                  video_length=68 // 3,
                                  # 2, 68, 3, 96, 96
                                  padding_mode='left',
                                  transforms=train_aug(image_size=96),
                                  n_skip=3)

    val_data = KERCVideoDataset(data_dir='/home/congvm/Dataset/',
                                csv_path='/home/congvm/Dataset/dataset/valid_faces.csv',
                                video_length=68,
                                padding_mode='left',
                                transforms=val_aug(image_size=96),
                                n_skip=3)

    # Dataloader
    data_loader = KERCVideoLoader(train_dataset=train_data,
                                  val_dataset=val_data,
                                  batch_size=8,
                                  num_workers=4)

    # Model
    model = get_VisionConvLSTM()

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
