import pprint
from argparse import ArgumentParser
import pytorch_lightning as pl
import functools
import warnings
import os
import torch
import numpy as np


def seed_everything(args):
    np.random.seed(args.seed)
    pl.seed_everything(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.cuda_deterministic
    torch.backends.cudnn.benchmark = args.cuda_benchmark


def load_args(is_notebook=False, verbose=True):
    parser = ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)
    parser = pl.LightningDataModule.add_argparse_args(parser)

    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--val_csv', default='data/valid_faces.csv', type=str)
    parser.add_argument('--train_csv', default='data/train_faces.csv', type=str)
    parser.add_argument('--test_csv', default='data/test_faces.csv', type=str)
    
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--weight_path', default='', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--freeze_backbone', action='store_true')

    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--aug_max_rot', default=20, type=int)
    parser.add_argument('--ckpt_prefix', default='', type=str)
    parser.add_argument('--tensorboard_dir', default='./', type=str)
    
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--cuda_benchmark', action='store_true')
    parser.add_argument('--cuda_deterministic', action='store_true')
    
    if is_notebook:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    if verbose:
        print('-' * 100)
        pprint.pprint(vars(args))
        print('-' * 100)
    return args


def deprecated(message=''):
    def deprecated_decorator(func):
        def deprecated_func(*args, **kwargs):
            warnings.warn("{} is a deprecated function. {}".format(func.__name__, message),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func(*args, **kwargs)
        return deprecated_func
    return deprecated_decorator
