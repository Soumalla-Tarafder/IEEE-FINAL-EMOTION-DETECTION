#from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from PIL import Image



# Define a flask app
app = Flask(__name__)
MODEL_PATH ='model1.h5'
# Load your trained model
model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    print(img_path)
    #img = image.load_img(img_path, target_size=(224, 224))
    img=cv2.imread(img_path)
    image_fromarray = Image.fromarray(img, 'RGB')
    resize_image = image_fromarray.resize((128, 128))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    x = input_data/255
   

    

    preds = model.predict(x)
    preds=preds.argmax()
    if preds==0:
        preds="EMOTION-->> Anger"
    elif preds==1:
        preds="EMOTION-->> Disgust"
    elif preds==2:
        preds="EMOTION-->> Fear"
    elif preds==3:
        preds="EMOTION-->> Happiness"
    elif preds==4:
        preds="EMOTION-->> Sadness"
    elif preds==5:
        preds="EMOTION-->> Surprise"
    
    
    
    return preds


@app.route('/', methods=['GET', 'POST'])
def index():
    # Main page
    return render_template('index.html')    



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)   