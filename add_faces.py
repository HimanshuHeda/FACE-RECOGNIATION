import cv2  # OpenCV for image and video processing
import pickle  # For saving/loading data
import numpy as np  # For array operations
import os  # For file and directory operations


# Start video capture from the default camera
video = cv2.VideoCapture(0)
# Load Haar cascade for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')


# List to store face data
faces_data = []

# Counter for frames
i = 0

# Get the name of the person from user input
name = input("Enter the name of the Person: ")

# Main loop to capture faces from video
while True:
    ret, frame = video.read()  # Read a frame from the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w :]  # Crop the face region
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize face to 50x50 pixels
        # Save every 10th frame, up to 100 faces
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i = i + 1  # Increment frame counter
        # Display number of faces collected on the frame
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Video", frame)  # Show the frame
    k = cv2.waitKey(1)
    # Exit loop if 'q' is pressed or 100 faces are collected
    if k == ord('q') or len(faces_data) == 100:
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()

# Convert faces_data list to numpy array and reshape for saving
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)


# Save or update names in name.pkl
if 'name.pkl' not in os.listdir('data/'):
    names = [name] * 100  # Create a list with the name repeated 100 times
else:
    with open('data/name.pkl', 'rb') as f:
        names = pickle.load(f)  # Load existing names
    names += [name] * 100  # Add new names
with open('data/name.pkl', 'wb') as f:
    pickle.dump(names, f)  # Save updated names

# Save or update faces data in faces_data.pkl
faces_data_path = 'data/faces_data.pkl'
if 'faces_data.pkl' in os.listdir('data/'):
    try:
        with open(faces_data_path, 'rb') as f:
            faces = pickle.load(f)  # Load existing faces data
        faces = np.append(faces, faces_data, axis=0)  # Append new faces data
    except (EOFError, pickle.UnpicklingError):
        faces = faces_data  # File is empty or corrupted, start fresh
else:
    faces = faces_data  # First entry
with open(faces_data_path, 'wb') as f:
    pickle.dump(faces, f)  # Save updated faces data



# cd /d "d:\Christ\5MCA\FACE RECOGNIATION"
# python add_faces.py