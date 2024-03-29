import numpy as np
import tensorflow as tf
import keras
import cv2
import os
import sys
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image
from scipy.special import softmax
import requests
from io import BytesIO
import io
import json

def get_embedding(face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    sample = np.expand_dims(face, axis=0)
    yhat = facenet_model.predict(sample)
    return yhat

def extract_faces_from_file(filename, required_size=(160, 160)):
    
    image = Image.open(filename)
    
    global image_data
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='PNG')
    image_data = imgByteArr.getvalue()
    
    image = image.convert('RGB')
    pixels = np.asarray(image)
    
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    
    faces = []
    
    for face in results:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        image = Image.fromarray(pixels[y1:y2, x1:x2])
        image = image.resize(required_size)
        faces.append((np.asarray(image), (x1, y1, x2, y2)))
    
    return faces


if os.path.isfile('./model/facenet_keras.h5'): 
    print("Loading model...")
    facenet_model = load_model('./model/facenet_keras.h5')
else:
    print("Model not found. Please download from github.com/anirudhajith/attendance-system.git")
    sys.exit(-1)

print("Generating vectors...")
people_list = os.listdir('./res/targets/')
old_people_list = os.listdir('./res/vectors/')
        
for index in range(len(people_list)):
        
    person = people_list[index]

    if person + ".npy" not in old_people_list:
        face = extract_faces_from_file(
            './res/targets/' + person + '/' + 
            os.listdir('./res/targets/' + person + '/')[0]
        )[0]
            
        vector = get_embedding(face[0])
        np.save('./res/vectors/' + person, vector)

    print('./res/vectors/' + person + ".npy")

