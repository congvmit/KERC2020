from mipkit import deprecated
from albumentations import Resize, Compose, ImageOnlyTransform
from albumentations import Rotate, HorizontalFlip
from albumentations import IAAAffine, ShiftScaleRotate
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2


# TRAIN_MEAN = [0.29685318, 0.23573298, 0.19259288]
# TRAIN_STD = [0.16447984, 0.14269175, 0.12425228]

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class BasicNormalize(ImageOnlyTransform):
    """Divide pixel values by 255
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(self, always_apply=False, p=1.0):
        super(BasicNormalize, self).__init__(always_apply=always_apply,
                                             p=p)

    def apply(self, image, **params):
        image = image / 255.0
        image = image.astype(np.float32)
        return image


@deprecated('Please use `video_train_aug`')
def train_aug(image_size, rot=45):
    return video_train_aug(image_size, rot)


@deprecated('Please use `video_val_aug`')
def val_aug(image_size):
    return video_val_aug(image_size)


def video_train_aug(image_size, rot=45):
    return Compose([
        Resize(image_size, image_size),
        BasicNormalize(),
        ToTensorV2()
    ])


def video_val_aug(image_size):
    return Compose([
        Resize(image_size, image_size),
        BasicNormalize(),
        ToTensorV2()
    ])

# ==================================================================


def image_multitask_train_aug(image_size, rot=45):
    return Compose([
        Resize(image_size, image_size),
        BasicNormalize(),
        ToTensorV2(),
    ])


def image_multitask_val_aug(image_size):
    return Compose([
        Resize(image_size, image_size),
        BasicNormalize(),
        ToTensorV2()
    ])


# ==================================================================
def image_train_aug(image_size, rot=45):
    return Compose([
        Resize(image_size, image_size),
        ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT,
                         value=0,
                         shift_limit=0.1),
        HorizontalFlip(),
        # Rotate(limit=rot),
        BasicNormalize(),
        ToTensorV2(),
    ])


def image_val_aug(image_size):
    return Compose([
        Resize(image_size, image_size),
        BasicNormalize(),
        ToTensorV2()
    ])
