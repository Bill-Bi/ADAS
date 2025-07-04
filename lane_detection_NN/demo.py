import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import cv2
import numpy as np
import tensorflow as tf 
import importlib

CONFIG_NAME = "config"
MODEL_PATH = "model_.025-0.022891.h5"
IMAGE_FOLDER = "../data/test/images"
OUTPUT_FOLDER = "test_results"

cfg = importlib.import_module(CONFIG_NAME)
tf.keras.backend.set_learning_phase(False)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Create output folder
import pathlib
pathlib.Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

img_paths = os.listdir(IMAGE_FOLDER)

for img_path in img_paths:

    if not (img_path.endswith("jpg") or img_path.endswith("png")):
        continue

    print(img_path)

    input_path = os.path.join(IMAGE_FOLDER, img_path)
    output_path = os.path.join(OUTPUT_FOLDER, img_path)
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_NEAREST)

    net_input = np.expand_dims(img, axis=0)
    predicted_mask = model.predict(net_input)[0]
    predicted_mask *= 255
    predicted_mask = predicted_mask.astype(np.uint8)
    predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)

    result = np.concatenate((img, predicted_mask), axis=1)
    cv2.imwrite(output_path, result)