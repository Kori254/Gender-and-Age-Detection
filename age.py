import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_model = load_model('models/age_model.h5')  #  age model
gender_model = load_model('models/gender_model.h5')  # The gender model

# Define mean values for image preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Define age ranges and gender classes
age_ranges = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_classes = ['Male', 'Female']

def preprocess_image(image):
    # Preprocess the image for gender and age prediction
    blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    return blob

def predict_age_gender(face):
    face_blob = preprocess_image(face)
    gender_model.setInput(face_blob)
    age_model.setInput(face_blob)
    
    gender_preds = gender_model.forward()
    age_preds = age_model.forward()
    
    gender = gender_classes[gender_preds[0].argmax()]
    age = age_ranges[age_preds[0].argmax()]
    
    return gender, age

def detect_and_predict(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        gender, age = predict_age_gender(face)
        
        label = f'{gender}, {age}'
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Age and Gender Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_image.jpg' with the path to your image
detect_and_predict('image/kori.jpg')
