import cv2
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import uuid

#Membuat Garis dan Titik
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
filename = "selamatpagi11.csv"
text_data = open((filename), "w")
text_data.write('Hasil'+'*')
data=[]

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
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
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

DATA_PATH = os.path.join('MP_Data')

actions = np.array(['hello','selamat','pagi','siang','sore','malam','sampaijumpa','selamat pagi','selamat siang','selamat sore','selamat malam'])

no_sequences = 10

sequence_length = 100

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

filename = 'Tes.mp4'
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    for action in actions :
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()

                image, results = mediapipe_detection(frame, holistic)
                pose_result = results.pose_landmarks
                lh_result = results.left_hand_landmarks
                rh_result = results.right_hand_landmarks

                if pose_result:
                    for id, lm in enumerate(pose_result.landmark):
                        #print('Pose :', id, lm)
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        #print('Pose:',
                              #['ID:', id,'Landmark:','X:', cx,'Y:', cy])
                        #print(id, cx, cy)
                if rh_result:
                    for id, lm in enumerate(rh_result.landmark):
                        #print('Right Hand :', id, lm)
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        #print('Right Hand:',
                              #['ID:', id,'Landmark:','X:', cx,'Y:', cy])
                        #print(id, cx, cy)
                if lh_result:
                    for id, lm in enumerate(lh_result.landmark):
                        #print('Right Hand :', id, lm)
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        #print('Left Hand:',
                              #['ID:', id,'Landmark:','X:', cx,'Y:', cy])
                        #print(id, cx, cy)


                im = draw_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('Indonesia Language Handtracking', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),1, cv2.LINE_AA)
                dr = r'C:\Users\CENTROCARE\PycharmProjects\IndonesianSignLanguageClassification\ImageOutput'
                #os.chdir(dr)
                #print("Before saving image:")
                #print(os.listdir(dr))
                filename = 'HandTracking {}.jpg'
                cv2.imwrite(filename, image)
                #print("After saving image:")
                #print(os.listdir(dr))
                #print('Successfully saved')

                keypoint = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoint)

                cv2.imshow("Indonesia Language Handtracking", image)

                #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                #plt.show()

                if cv2.waitKey(10) & 0xff == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
