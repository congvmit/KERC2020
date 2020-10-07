python main_images.py \
    --seed 2020 \
    --log_save_interval 1 \
    --is_training \
    --gpus 1 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --num_workers 6 \
    --max_epochs 20 \
    --data_dir /home/congvm/Dataset/ \
    --train_csv /home/congvm/Dataset/dataset/train_faces.csv \
    --val_csv /home/congvm/Dataset/dataset/valid_faces.csv \
    --ckpt_prefix resnet50 \
    --tensorboard_dir logs

# --accumulate_grad_batches 16 \
# --cuda_deterministic \
# --cuda_benchmark \
# --freeze_backbone
# --load_weights /home/congvm/Workspace/CNU2020/Research/emotion/source/emo/weights/mobilenet_v2-b0353104.pth
