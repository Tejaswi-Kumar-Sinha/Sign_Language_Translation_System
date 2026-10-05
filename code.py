#python -m pip install opencv-python mediapipe tensorflow scikit-learn matplotlib
#importing dependencies
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic #Holistic model:to make detections
mp_drawing = mp.solutions.drawing_utils #Drawing Utilities:To draw them
#media pipe detection function 
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)#Color conversion BGR 2 RGB
    image.flags.writeable = False #Image is no longer writeable 
    results = model.process(image)#make prediction i.e it detects
    image.flags.writeable = True#Image is now writeable
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)#Color conversion RGB 2 BGR
    return image,results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) #Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw right hand connections

    mp_holistic.POSE_CONNECTIONS
#using opencv
cap = cv2.VideoCapture(0)#accessing webcam
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as holistic:
 while cap.isOpened():#creating a loop for mutliple frames
    #Read feed
    ret, frame = cap.read()

    #Make detections
    image,results=mediapipe_detection(frame,holistic)
    print(results)

    #Draw landmarks
    draw_landmarks(image,results)

    #show to screen
    cv2.imshow('OpenCV Feed', image)#Can change OpenCV Feed:Just a name of interface

    #break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):#press q to quit
        break
cap.release()
cv2.destroyAllWindows()