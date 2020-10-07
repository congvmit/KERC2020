
import os
import torch
import numpy as np
from train_base import Trainer


class KERC2020(Trainer):
    def experiment_setup(self, args):
        from dataloader import KERCImageDataset, KERCDataLoader
        from modeling import load_resnet50
        from augmentation import image_train_aug, image_val_aug

        train_data = KERCImageDataset(data_dir=args.data_dir,
                                      csv_path=args.train_csv,
                                      transforms=image_train_aug(image_size=args.image_size,
                                                                 rot=args.aug_max_rot))

        val_data = KERCImageDataset(data_dir=args.data_dir,
                                    csv_path=args.val_csv,
                                    transforms=image_val_aug(image_size=args.image_size))

        # Dataloader
        dataloader = KERCDataLoader(train_dataset=train_data,
                                    val_dataset=val_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    gpus=args.gpus)

        # Model
        model = load_resnet50(learning_rate=args.learning_rate)

        model.example_input_array = torch.zeros([1,
                                                 3,
                                                 args.image_size,
                                                 args.image_size])

        if args.weight_path:
            print('Loading weights from {}'.format(args.weight_path))
            state_dict = torch.load(args.weight_path)
            model.load_state_dict(state_dict, strict=False)
        return args, dataloader, model


if __name__ == '__main__':
    KERC2020.start()
