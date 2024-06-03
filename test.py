import cv2
import face_recognition
import os
import numpy as np

# Load the known images and encode the faces
def load_known_faces(known_faces_dir):
    known_faces = []
    known_names = []
    for name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, name)
        for filename in os.listdir(person_dir):
            image_path = os.path.join(person_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
    return known_faces, known_names

# Initialize the known faces and names
known_faces_dir = "dataset"
known_faces, known_names = load_known_faces(known_faces_dir)

# Function to recognize faces in an image
def recognize_faces_in_image(image):
    # Convert the image from BGR (OpenCV) to RGB (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]

        # Draw a box around the face and label it with the name
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Capture video from the webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Recognize faces in the frame
    frame = recognize_faces_in_image(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
video_capture.release()
cv2.destroyAllWindows()
