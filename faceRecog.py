import cv2
import os
import numpy as np


# This module contains all common functions that are called in tester.py file


# Given an image below function returns rectangle for face detected along with gray scale image
def face_detection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    # convert color image to gray scale
    face_haar_cascade=cv2.CascadeClassifier('/Users/areeb/PycharmProjects/test1/haarCascade/haar_Cascade_frontFace'
                                            '_default.xml')
    # Load haar classifier for detecting images
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.34, minNeighbors=5)
    # detectMultiScale returns rectangles..... image resizing

    return faces, gray_img
# labels for faces using os.walk which will return subdirectory, path and filename


def labels_for_training_data(directory):
    faces = []
    faceID = []
    for path, subDirNames, fileNames in os.walk(directory):
        for fileName in fileNames:
            if fileName.startswith("."):
                print("Skipping system file")  # Skipping files that start with '.'
                continue

            id = os.path.basename(path)  # fetching subdirectory names
            img_path = os.path.join(path, fileName) # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img=cv2.imread(img_path) # loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = face_detection(test_img)
            # Calling face_detection function to return faces detected in particular image
            if len(faces_rect) != 1:
                continue  # Since we are assuming only single person images are being fed to classifier
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y+w, x:x+h] # Selecting the region of interest(face) from the gray image
            # Croping the part of face from the image
            faces.append(roi_gray)
            faceID.append(int(id)) # The classifier can only take int datatype
    return faces, faceID


# Training the classifier to detect the face
def train_classifier(faces, faceID):
    # Local Binary Pattern Histogram
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID)) # The recognizer takes in the labels as numpy array
    return face_recognizer


# Drawing the box around the face
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


# Putting the text with the rectangle box
def put_text(test_img, text, x, y):
    # Image, text, image co-ordinates, font-style, font-weight, colour, font-size
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 3)
