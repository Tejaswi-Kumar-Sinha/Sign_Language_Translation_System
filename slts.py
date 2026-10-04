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
#media pipe draw function
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) #Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #Draw right hand connections
#using media pipe and stylising the drawings
def draw_styled_landmarks(image,results):
   mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                            mp_drawing.DrawingSpec(color=(80,110,10),thickness=1, circle_radius=1),
                            mp_drawing.DrawingSpec(color=(80,256,121),thickness=1, circle_radius=1))
    #Draw face connections
   mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10),thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121),thickness=2, circle_radius=2))
    #Draw pose connections
   mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76),thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250),thickness=2, circle_radius=2))
    #Draw left hand connections
   mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66),thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230),thickness=2, circle_radius=2))
    #Draw right hand connections
   
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
    draw_styled_landmarks(image,results)

    #show to screen
    cv2.imshow('Input', image)#Can change OpenCV Feed:Just a name of interface

    #break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):#press q to quit
        break

#to extract keypoints from the results of mediapipe detection
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

#path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data')#creating a folder to store data
#actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])#creating a numpy array of actions
#number of sequences and videos for each action
no_sequences = 30
#Videos are going to be 30 frames in length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)#accessing webcam
#set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5 , min_tracking_confidence=0.5) as holistic:

  #New Loop
  #loop through actions
  for action in actions:
    #loop through sequences aka videos
    for sequence in range(no_sequences):
      #loop through video length aka sequence length
      for frame_num in range(sequence_length):
         #Read feed
               ret, frame = cap.read()
         
               #Make detections
               image,results=mediapipe_detection(frame,holistic)                
               print(results)
         
               #Draw landmarks
               draw_styled_landmarks(image,results)
         
               #New Apply wait logic
               if frame_num == 0:
                 cv2.putText(image, 'STARTING COLLECTION', (120,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                 #show to screen
                 cv2.imshow('Input', image)#Can change OpenCV Feed:Just a name of interface
                 cv2.waitKey(2000)
               else:
                                 cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                                 #show to screen
                                 cv2.imshow('Input', image)#Can change OpenCV Feed:Just a name of interface
         
               #New Export keypoints
               keypoints = extract_keypoints(results)
               npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
               np.save(npy_path, keypoints)
         
               #break gracefully
               if cv2.waitKey(10) & 0xFF == ord('q'):#press q to quit
                break
         