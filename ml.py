import cv2
from deepface import DeepFace

def detect_and_predict(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Convert the image to gray scale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = image[y:y+h, x:x+w]
        
        # Predict age and gender using DeepFace
        predictions = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)
        
        # Extract predictions
        age = predictions['age']
        gender = predictions['gender']
        
        # Prepare label
        label = f'{gender}, {age}'
        
        # Draw rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Put label near the face
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the output
    cv2.imshow('Age and Gender Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'path_to_image.jpg' with the path to your image
detect_and_predict('image/kori.jpg')
