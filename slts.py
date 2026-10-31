#python -m pip install opencv-python mediapipe tensorflow scikit-learn matplotlib
#importing dependencies
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import kagglehub
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

#to extract keypoints from the results of mediapipe detection
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Automatically download the keypoint dataset from Kaggle
kaggle_path = kagglehub.dataset_download("swaptr/indian-sign-language-mediapipe-holistic-landmarks") # Replace with your specific Kaggle dataset slug/path

# path for exported data, numpy arrays pointing directly to the Kaggle folder
DATA_PATH = os.path.join(kaggle_path) 

# Automatically grab the folder names from the dataset and take the first 3 for quick training
all_actions = np.array([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
actions = all_actions[:3] 
print(f"Dynamic Actions Loaded: {actions}")

#number of sequences and videos for each action
no_sequences = 30
#Videos are going to be 30 frames in length
sequence_length = 30

label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequences_in_action = [d for d in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, d))][:no_sequences]
    
    for sequence in sequences_in_action:
        window = []
        seq_path = os.path.join(action_path, sequence)
        for frame_num in range(sequence_length):
            npy_file = os.path.join(seq_path, "{}.npy".format(frame_num))
            if os.path.exists(npy_file):
                res = np.load(npy_file)
            else:
                res = np.zeros(1662) # Fallback vector
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

X = np.array(sequences)
y = to_categorical(labels).astype(int)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

#Build and train the LSTM neural network
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=30, callbacks=[tensorboard_callback]) #To train the model
#Make predictions
res = model.predict(x_test)
#save weights
model.save('action.h5')
del model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.load_weights('action.h5')
#Valuate the model
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
print(accuracy_score(ytrue, yhat))

#New detection variables
sequence = []   
sentence = []
predictions = []
treshold = 0.4

#using OpenCV
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

    #Prediction logic
    keypoints = extract_keypoints(results)
    sequence.insert(0, keypoints)
    sequence = sequence[:30]

    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
       # print(res)
        print(actions[np.argmax(res)])
        predictions.append(np.argmax(res))

        # viz logic
        if np.unique(predictions[-10:])[0]==np.argmax(res):   
            if res[np.argmax(res)] > treshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])
                if len(sentence) > 5: 
                   sentence = sentence[-5:]

    #Viz probabilities
   # image = prob_viz(res, actions, image, colors)

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #show to screen
    cv2.imshow('Input', image)#Can change OpenCV Feed:Just a name of interface

    #break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):#press q to quit
        break
#abc
