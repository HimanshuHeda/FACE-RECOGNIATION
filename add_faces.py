import cv2
import pickle

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []

i = 0

name = input("Enter the name of the Person: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len (faces_data) <= 100 and i%10 == 0:
            faces_data.append(resized_img)
        i=i+1
        cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:  # Press 'q' to exit
        break

video.release()
cv2.destroyAllWindows()