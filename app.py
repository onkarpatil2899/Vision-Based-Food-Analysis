from http.client import PRECONDITION_FAILED
from tkinter import Y
from flask import Flask, render_template, request
import numpy as np
from skimage.transform import resize
import pickle
from PIL import Image
import csv
import requests
import pandas as pd
from werkzeug.utils import secure_filename
from tensorflow import keras
from skimage.transform import resize
import itertools
import os

import matplotlib.pylab as plt
import numpy as np

import tensorflow as tf


from skimage.io import imread
import matplotlib.pyplot as plt

import recipe.recipe as r

# load the model

model = keras.models.load_model('model')
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/new_dish')
def new_dish():
    return render_template('new_dish.html')



@app.route('/tracking')
def tracking():
    return render_template('tracking.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/logged')
def logged():
    return render_template('logged.html')





@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefileID']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)
    #class_names=['biryani', 'burger', 'cupcakes', 'dosa', 'icecream', 'idli', 'paratha', 'pizza', 'samosa', 'soup', 'vadapav']
    class_names=('biryani', 'burger', 'cupcakes', 'dosa', 'icecream', 'idli', 'paratha', 'pizza', 'samosa', 'soup', 'vadapav')
    image = Image.open(imagefile)
    image = np.array(image)
    img_resized = resize(image, (512, 512, 3))
    image = img_resized
    prediction_scores = model.predict(np.expand_dims(image, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + class_names[predicted_index])
    y_out=class_names[predicted_index]
    #print("Predicted label: " + class_names[predicted_index])
    global prediction
    prediction = y_out
    return render_template('new_dish.html', prediction=y_out)


@app.route('/nutri')
def nutri():
    nutri_value = prediction
    return render_template('nutri.html', nut=nutri_value)

@app.route('/recipe', methods=['GET'])
def recipe():
    payload = {
        "recipe": r.food_type[prediction.lower()]
    }

    print(payload)

    return render_template('recipe.html', recipe = payload)


@app.route('/recipe_ingredients/<number>', methods=['GET'])
def recipe_ingredients(number):
    print(number)

    payload = {
        "ingredients": r.food_ingredients[str(number)]
    }

    return render_template('ingredients.html', payload = payload)

   




if __name__ == '__main__':
    app.run(port=3000, debug=True)
