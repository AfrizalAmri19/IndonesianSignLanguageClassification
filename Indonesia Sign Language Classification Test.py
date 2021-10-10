import cv2
import numpy as np
from tensorflow.keras.models import Sequential
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import uuid

model = Sequential
model.load_weights('IndonesiaSignLanguage.h5')

#Membuat Garis dan Titik
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 150, 150), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(150, 150, 0), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(100, 250, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 150, 255), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(180, 150, 0), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(100, 150, 20), thickness=2, circle_radius=2)
                              )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

actions = np.array(['hello','selamat','pagi','siang','sore','malam','sampaijumpa'])

sequence = []
sentence = []
threshold = 0.8

filename = 'Tes.mp4'
cap = cv2.VideoCapture(filename)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        ret, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)

        im = draw_landmarks(image, results)

        keypoint = extract_keypoints(results)
        sequence.append(keypoint)
        sequence = sequence[-30:]
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence,axis=0))[0]
            print(actions[np.argmax(res)])



        cv2.imshow("Indonesia Language Handtracking", image)

        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.show()

        if cv2.waitKey(10) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()