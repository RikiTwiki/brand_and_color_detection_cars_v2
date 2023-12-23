from fastai import *
from fastai.vision.all import *
from fastai.metrics import error_rate
import os
from keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from fastai.vision.all import *
from PIL import Image

data = ImageDataLoaders.from_folder('../imgs_zip/imgs/train', train='.', valid_pct=0.2, 
                                   size=224, item_tfms=Resize(224))
learn = cnn_learner(data, models.resnet18, model_dir='')
learn.export(Path("export.pkl"))
learn.model_dir = ""
learn = load_learner('export.pkl')
learn.load("stage-1_v2")

def predict_image(image_path):
    # Open the image
    img = Image.open(image_path)

    # Resize the image (if necessary)
    img_resized = img.resize((224, 224))

    # Predict
    pred_class, pred_idx, probs = learn.predict(img_resized)

    # Print results
    print(f"Predicted class: {pred_class}")
    print(f"Predicted probabilities: {probs[pred_idx]:.4f}")

    # Display the image
    img_resized.show()

# Example usage
predict_image('test_images/e30b8ff074b04f392dedbf49da9a8f28_640x480.jpg')
