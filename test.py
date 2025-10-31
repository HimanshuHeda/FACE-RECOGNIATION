
import csv
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


# Load labels and faces
with open('data/name.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure FACES and LABELS have the same number of samples
min_len = min(len(LABELS), FACES.shape[0])
LABELS = LABELS[:min_len]
FACES = FACES[:min_len]

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['NAME', 'DATE', 'TIME']


# Set to keep track of names already marked present in this session
marked_names = set()

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    recognized_names = set()
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        name = str(output[0])
        # If the predicted name is not in the known label list, mark as unknown
        if name not in set(LABELS):
            display_name = "Unknown User"
        else:
            display_name = name
            recognized_names.add(name)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, display_name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    imgBackground[162:162+480, 55:55+640] = frame
    cv2.imshow("Video", imgBackground)
    k = cv2.waitKey(1)
    if k == ord('o'):
        # Only mark attendance for new recognized names in this session
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        attendance_file = f"Attendance/Attendance_{date}.csv"
        os.makedirs('Attendance', exist_ok=True)
        file_exists = os.path.isfile(attendance_file)
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(COL_NAMES)
            for name in recognized_names:
                if name not in marked_names:
                    writer.writerow([name, date, timestamp])
                    marked_names.add(name)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()