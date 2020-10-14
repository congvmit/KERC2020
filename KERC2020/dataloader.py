import warnings
import cv2
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

import pandas as pd
import os
from PIL import Image
from typing import Sequence
from augmentation import to_categorical


class AffectNetDataset(Dataset):
    def __init__(self, data_dir: str,
                 csv_path: str,
                 transforms: transforms.Compose = None, args=None):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.transforms = transforms
        self.args = args

    def __getitem__(self, idx):
        # 'subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal'
        data = self.data_df.iloc[idx]

        image_path = data.subDirectory_filePath
        image_path = os.path.join(self.data_dir, image_path)

        expression = data.expression
        expression = torch.as_tensor(expression, dtype=torch.long)

        img_arr = cv2.imread(image_path)[..., ::-1]

        img_h, img_w, img_c = img_arr.shape

        # Crop face
        x = int(data.face_x)
        y = int(data.face_y)
        w = int(data.face_width)
        h = int(data.face_height)

        face_arr = img_arr[y:y + h, x:x + w, ...]

        if self.transforms:
            face_arr = self.transforms(image=face_arr)['image']

        # Load valence, arousal
        # -1 1
        valence = torch.as_tensor([data.valence], dtype=torch.float32)
        arousal = torch.as_tensor([data.arousal], dtype=torch.float32)

        # Load landmarks
        landmarks = np.array(list(map(float, data.facial_landmarks.split(';'))))
        landmarks = landmarks.reshape((68, 2))
        landmarks[:, 0] /= img_w
        landmarks[:, 1] /= img_h
        landmarks = torch.from_numpy(landmarks.flatten()).float()

        return {
            'image': face_arr,  # img should be a tensor
            'expression': expression,
            'valence': valence,
            'arousal': arousal,
            'landmarks': landmarks
        }

    def __len__(self):
        return len(self.data_df)


class KERC2019Dataset(Dataset):
    def __init__(self, data_dir: str,
                 csv_path: str,
                 transforms: transforms.Compose = None):
        self.convert_dict = {'Angry': 0,
                             'Disgust': 1,
                             'Fear': 2,
                             'Happy': 3,
                             'Neutral': 4,
                             'Sad': 5,
                             'Surprise': 6}
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, idx):
        # image_paths, labels
        data = self.data_df.iloc[idx]

        image_path = data.image_paths
        image_path = os.path.join(self.data_dir, image_path)

        label = data.labels

        label = self.convert_dict[label]
        label = torch.as_tensor(label, dtype=torch.long)

        img_arr = cv2.imread(image_path)[..., ::-1]
        if self.transforms:
            img_arr = self.transforms(image=img_arr)['image']

        return {
            'image': img_arr,  # img should be a tensor
            'label': label
        }

    def __len__(self):
        return len(self.data_df)


class KERCImageDataset(Dataset):
    def __init__(self, data_dir: str,
                 csv_path: str,
                 transforms: transforms.Compose = None):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, idx):
        # video_name, frame_files, face_files, valence, arousal, stress
        data = self.data_df.iloc[idx]
        face_file = data.face_files
        face_file = face_file
        img_path = os.path.join(self.data_dir, face_file)

        valence = data.valence / 10.
        arousal = data.arousal / 10.
        stress = data.stress / 10.

        valence = torch.as_tensor([valence], dtype=torch.float32)
        arousal = torch.as_tensor([arousal], dtype=torch.float32)
        stress = torch.as_tensor([stress], dtype=torch.float32)

        img_arr = cv2.imread(img_path)[..., ::-1]
        if self.transforms:
            img_arr = self.transforms(image=img_arr)['image']

        return {
            'image': img_arr,  # img should be a tensor
            'arousal': arousal,
            'valence': valence,
            'stress': stress,
        }

    def __len__(self):
        return len(self.data_df)


class KERCVideoDataset(Dataset):
    def __init__(self, data_dir,
                 csv_path,
                 video_length=0,
                 padding_mode='left',
                 transforms=None,
                 verbose=1, n_skip=1):
        self.n_skip = n_skip
        self.data_dir = data_dir
        data_df = pd.read_csv(csv_path, sep=",")
        self.data_df = self._prepare_frames(data_df)
        self.verbose = verbose
        self.transforms = transforms
        self.padding_mode = padding_mode
        self.video_length = video_length

    def _prepare_frames(self, data_df):
        # video_name, frame_files, face_files, valence, arousal, stress
        video_files = np.unique(data_df.video_name)
        print('Preparing video for KERC Video Dataset!')
        df = pd.DataFrame([], columns=['video_name', 'face_lists', 'valence', 'arousal', 'stress'])
        for i, video_name in tqdm(enumerate(video_files), total=len(video_files)):
            select_data_df = data_df[data_df.video_name == video_name]
            df.loc[i] = {'video_name': video_name,
                         'face_lists': select_data_df.face_files,
                         'valence': select_data_df.iloc[0].valence,
                         'arousal': select_data_df.iloc[0].arousal,
                         'stress': select_data_df.iloc[0].stress
                         }
        return df

    def _padding_video(self,
                       list_arr: Sequence[torch.Tensor],
                       padding_mode: str,
                       video_length: int):
        ''' Add more frames to a squence of image array corresponding to sample_length
            If the length of the list_arr is smaller than sampling_length, 
        '''
        if padding_mode == 'naive':
            raise NotImplementedError

        elif padding_mode == 'last':
            raise NotImplementedError

        elif padding_mode == 'left':
            # Padding zeros at the begining of the video
            padding_width = video_length - len(list_arr)

            # Get size of the first frame
            c, h, w = list_arr[0].shape
            if padding_width > 0:
                for i in range(padding_width):
                    pad = torch.zeros((c, h, w), dtype=torch.float32)
                    list_arr.insert(0, pad)

            elif padding_width < 0:
                sampled_list_arr = []
                # Sample `video_length` frames
                indexes = np.random.choice(range(0, len(list_arr)),
                                           size=video_length,
                                           replace=False)
                for i in indexes:
                    sampled_list_arr.append(list_arr[i])
                list_arr = sampled_list_arr.copy()
            return list_arr
        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        # video_name, frame_files, face_files, valence, arousal, stress
        data = self.data_df.iloc[idx, :]
        face_lists = data.face_lists
        list_img = []
        for i in range(0, len(face_lists), self.n_skip):
            img_path = face_lists.iloc[i]
            img_path = os.path.join(self.data_dir, img_path)
            img_arr = cv2.imread(img_path)[..., ::-1]

            if self.transforms:
                img_arr = self.transforms(image=img_arr)['image']
            list_img.append(img_arr)

        list_img = self._padding_video(list_img,
                                       self.padding_mode,
                                       self.video_length)

        if self.transforms:
            list_img = torch.stack(list_img)

        valence = torch.as_tensor([data.valence], dtype=torch.float32)
        arousal = torch.as_tensor([data.arousal], dtype=torch.float32)
        stress = torch.as_tensor([data.stress], dtype=torch.float32)

        return {
            'image': list_img,  # img should be a tensor
            'arousal': arousal,
            'valence': valence,
            'stress': stress
        }

    def __len__(self):
        return len(self.data_df)


class KERCDataLoader(pl.LightningDataModule):

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


class KERCVideoLoader(KERCDataLoader):
    def __init__(self, *args, **kwargs):
        warnings.warn("The 'KERCVideoLoader' class was renamed 'KERCDataLoader'",
                      DeprecationWarning, stacklevel=2)

        super().__init__(*args, **kwargs)
