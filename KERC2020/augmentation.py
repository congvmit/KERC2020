from albumentations import *
from albumentations.pytorch import ToTensorV2


TRAIN_MEAN = [0.29685318, 0.23573298, 0.19259288]
TRAIN_STD = [0.16447984, 0.14269175, 0.12425228]


def train_aug(image_size, rot=45):
    return Compose([
        Resize(image_size, image_size),
        HorizontalFlip(),
        Rotate(limit=rot),
        Normalize(TRAIN_MEAN, TRAIN_STD),
        ToTensorV2()
    ])


def val_aug(image_size):
    return Compose([
        Resize(image_size, image_size),
        Normalize(TRAIN_MEAN, TRAIN_STD),
        ToTensorV2()
    ])
