import losses

description = "Training with Bill's configuration"

model_type = "UNET"

CLASSES = {'lane_line': (255, 255, 255)}
LR = 0.001
EPOCHS = 50
BATCH_SIZE = 20

SAVE_ALL_MODELS = True

image_size = 384

# Data folder for training, validation and testing
list_x_train_dirs = [
    '../data/train/images'
]
list_y_train_dirs = [
    '../data/train/masks'
]
list_x_valid_dirs = [
    '../data/val/images'
]
list_y_valid_dirs = [
    '../data/val/masks'
]
list_x_test_dir = [
    '../data/test/images'
]
list_y_test_dir = [
    '../data/test/masks'
]

# All backbone option
# 'vgg16' 'vgg19'  
# 'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
# 'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
# 'resnext50' 'resnext101'
# 'seresnext50' 'seresnext101'
# 'senet154'
# 'densenet121' 'densenet169' 'densenet201'
# 'inceptionv3' 'inceptionresnetv2'
# 'mobilenet' 'mobilenetv2'
# 'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'

backbone_name='resnet18'
input_shape = (image_size, image_size, 3)
encoder_freeze = False
encoder_weights = 'imagenet'
encoder_features = 'default'
decoder_block_type = 'transpose'    # 'upsampling or 'transpose'
decoder_filters = (64, 64, 32, 16, 8)

import os
x = os.path.basename(__file__).replace('.py', '')

import pathlib

base_log_dir = "saved_models/{}".format(x)
pathlib.Path(base_log_dir).mkdir(parents=True, exist_ok=True)

base_model_path = os.path.join(base_log_dir, "model_")
log_file = os.path.join(base_log_dir, "log.csv")
my_log_file = os.path.join(base_log_dir, "mylog.csv")
saved_freeze_model = os.path.join(base_log_dir, "freezed_model.pb")
tensorboard_log_dir = os.path.join(base_log_dir, "tb_logs")

loss = losses.binary_crossentropy

# threshold and per_image for evaluation metrics
metric_threshold = 0.5
metric_per_image = True