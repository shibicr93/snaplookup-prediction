import os

import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory,redirect
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import json
import urllib.request

app = Flask(__name__)
classes = ["Bathroom Mirrors", "Bathroom Shelves", "Bathroom Sinks", "Ceiling Fans", "Curtain Rods", "Pet Beds", "Rugs", "Toilet Paper Holders", "Towel Bars", "Towel Hooks", "Towel Racks", "Towel Rings"]
transfer_learning_model = tf.keras.models.load_model("snaplookupv2.h5")

# home page
@app.route("/")
def home():
  return render_template("test.html")
                  

@app.route('/predict',methods=['POST'])
def predict():
    file = request.files['imgfile']
    preprocessed_image=tf.image.decode_jpeg(file.read(),channels=3)
    preprocessed_image=tf.image.resize(preprocessed_image,[100,100])
    preprocessed_image/=255.0
    preprocessed_image=tf.reshape(preprocessed_image,(1,100,100,3));
    res = transfer_learning_model.predict_classes(preprocessed_image, 1, verbose=0)[0]
    print("response:",res)
    print("response:",classes[res])
    return "https://www.lowes.com/search?searchTerm="+classes[res], 200

if(__name__=='__main__'):
  app.run(port=8080,debug='true',threaded=False)

