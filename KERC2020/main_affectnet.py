import torch
from base import Trainer


class AffectNetTrainer(Trainer):
    def experiment_setup(self, args):
        from dataloader import AffectNetDataset
        from base import BaseDataLoader
        from modeling.classsifier import load_affectnet_resnet50
        from augmentation import image_multitask_train_aug, image_multitask_val_aug

        train_data = AffectNetDataset(data_dir=args.data_dir,
                                      csv_path=args.train_csv,
                                      transforms=image_multitask_train_aug(image_size=args.image_size,
                                                                           rot=args.aug_max_rot))

        val_data = AffectNetDataset(data_dir=args.data_dir,
                                    csv_path=args.val_csv,
                                    transforms=image_multitask_val_aug(image_size=args.image_size))

        # Dataloader
        dataloader = BaseDataLoader(train_dataset=train_data,
                                    val_dataset=val_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    gpus=args.gpus)

        # Model
        if args.backbone == 'resnet50':
            model = load_affectnet_resnet50(learning_rate=args.learning_rate,
                                            dropout_rate=args.dropout_rate)
        else:
            raise ValueError('Unknown network')

        model.example_input_array = torch.zeros([1,
                                                 3,
                                                 args.image_size,
                                                 args.image_size])

        if args.weight_path:
            print('Loading weights from {}'.format(args.weight_path))
            state_dict = torch.load(args.weight_path)
            print(model.load_state_dict(state_dict['state_dict'],
                                        strict=False))

        if args.freeze_backbone:
            print('Freezing backbone.')
            for param in model.backbone.parameters():
                param.requires_grad = False

        return args, dataloader, model


if __name__ == '__main__':
    AffectNetTrainer().start()
