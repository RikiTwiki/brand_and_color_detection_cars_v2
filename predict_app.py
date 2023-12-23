import base64
import numpy as np
import io
from PIL import Image
from flask import request
from flask import jsonify
from flask import Flask
import json
from pathlib import Path
from skimage import transform
import tensorflow.compat.v1 as tf

import fastai
from fastai import *
from fastai.vision.all import *
from fastai.callback import *

app = Flask(__name__)

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# def get_model():
learn = load_learner('F:\projects\Image-classification-fastai\export.pkl')

print("  * loading keras model...")
# get_model()

# @app.route("/predict", method=['POST'])
@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded))

    img_resized = img.resize((224, 224))
    pred, pred_idx, probs = learn.predict(img_resized)

    print("pred")
    print("probs[pred_idx]")

    response = {
        'prediction' : {
            'output' : str(pred),
            'probability' : str(probs[pred_idx])
            # ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
            # 'buildings': str(pred[0]),
            # 'forest': str(pred[1]),
            # 'glacier': str(pred[2]),
            # 'mountain': str(pred[3]),
            # 'sea': str(pred[4]),
            # 'street': str(pred[5])
        }
    }

    return jsonify(response)

# link of localhost
# http://localhost:5000/static/predict.html

sess = tf.Session()
tf.compat.v1.keras.backend.set_session(sess)
graph = tf.get_default_graph()

def predict():
    global sess
    global graph
    with graph.as_default():
        message = request.get_json(force=True)