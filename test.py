from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle
import numpy as np
import os
import time 
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/name.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'DATE', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 1) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
        cv2.rectangle(frame, (x,y-40), (x+w,y), (50,50,255), -1)
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2) 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        # Create a proper attendance string instead of using subscript on str
        # Format: NAME, DATE, TIME
        name = str(output[0])
        attendence = f"{name},{date},{timestamp}"
        # Optionally append attendance to a CSV file (creates file if missing)
        attendance_path = os.path.join('data', 'attendance.csv')
        # Ensure the data folder exists
        os.makedirs('data', exist_ok=True)
        with open(attendance_path, 'a') as af:
            af.write(attendence + "\n")
    imgBackground[162:162+480, 55:55+640] = frame
    cv2.imshow("Video", imgBackground)
    k = cv2.waitKey(1)
    if k == ord('q'):  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()