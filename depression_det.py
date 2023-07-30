import cv2
import numpy as np

from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array

# Get the prediction
model = load_model("C:/Users/dhana/OneDrive/Desktop/depression/model30.h5")
face_detector1 = cv2.CascadeClassifier('C:/Users/dhana/OneDrive/Desktop/depression/haarcascade_frontalface_default.xml')

# reading the input image now
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_detector1.detectMultiScale(gray,1.1, 4 )
    for (x,y, w, h) in faces:
        
        img = img_to_array(frame[y:y + h, x:x + w])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48,48), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        class_ = np.argmax(model.predict(img))
        if class_==0:
            # print("Depressed")
            label_ = "Depressed"
        else:
            label_ = "Not Depressed"
            
        rect = cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
        cv2.putText(rect, label_, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv2.imshow("window", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break























































