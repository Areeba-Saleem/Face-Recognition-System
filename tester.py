import cv2
import os
import numpy as np
import faceRecog as fr

# This module takes images  stored in disk and performs face recognition
test_img = cv2.imread('/Users/areeb/PycharmProjects/test1/testImages/areeba1.jpg') # test_img path
faces_detected, gray_img = fr.face_detection(test_img) # detect faces
print("faces_detected:", faces_detected)

# # Drawing rectangle around the detected faces
# for (x, y, w, h) in faces_detected:
#     cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), thickness=5)

# resized_img = cv2.resize(test_img, (1000, 700))
# cv2.imshow("face detection: ", resized_img)
# cv2.waitKey(0) # wait until key press
# cv2.destroyAllWindows()

# faces, faceID = fr.labels_for_training_data('/Users/areeb/PycharmProjects/test1/trainingImages')
# face_recognizer = fr.train_classifier(faces, faceID)
# face_recognizer.save('trainingData.yml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('/Users/areeb/PycharmProjects/test1/trainingData.yml')
name = {0: "Areeba",
        1: "Iqra",
        2: "Tanzila"}

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y+h, x:x+w]
    # return the label name and the confidence value of the image
    label, confidence = face_recognizer.predict(roi_gray)
    print("Confidence: ", confidence)
    print("Label: ", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if confidence > 60:
        continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 700))
cv2.imshow("face detection: ", resized_img)
cv2.waitKey(0) # wait until key press
cv2.destroyAllWindows()
