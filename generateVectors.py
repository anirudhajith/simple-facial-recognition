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

def get_embedding(face_image_array):
    face_image_array = face_image_array.astype('float32')
    mean, std = face_image_array.mean(), face_image_array.std()
    face_image_array = (face_image_array - mean)/std
    sample = np.expand_dims(face_image_array, axis=0)
    embedding = facenet_model.predict(sample)
    return embedding

def extract_face_from_file(filename, required_size=(160, 160)):
    
    image = Image.open(filename).convert('RGB')
    pixels = np.asarray(image)
    
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    face_data_item = results[0]
    
    x1, y1, width, height = face_data_item['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face_image = Image.fromarray(pixels[y1:y2, x1:x2]).resize(required_size)   

    return np.asarray(face_image)


if os.path.isfile('./model/facenet_keras.h5'): 
    print("Loading model...")
    facenet_model = load_model('./model/facenet_keras.h5')
else:
    print("Model not found. Please download from github.com/anirudhajith/attendance-system.git")
    sys.exit(-1)

print("Generating vectors...")
people_set = set(os.listdir('./res/targets/'))
old_people_set = set(filename.split(".")[0] for filename in os.listdir('./res/vectors/'))
new_people_set = people_set - old_people_set
        
for person in new_people_set:
    detected_face = extract_faces_from_file(
        './res/targets/' + person + '/' + 
        os.listdir('./res/targets/' + person + '/')[0]
    )
    face_embedding = get_embedding(detected_face)
    np.save('./res/vectors/' + person, face_embedding)

    print('Added ./res/vectors/' + person + ".npy")