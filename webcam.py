import cv2
from keras.models import model_from_json
import numpy as np

label = {0:"Angry", 1:"Disgusted", 2:"Fearful", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}

##load json and create file from our emotiondetection ipnby file
json_file = open('model.json')
loaded_json = json_file.read()
json_file.close()
model = model_from_json(loaded_json)      #conert to model

##load weights into new model
model.load_weights('model.h5')
print("Model is Loaded.")


##webcame
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame, (1280,720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##detect num of faces available in camera
    nfaces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    #preprocessing each face from the cam
    for (x,y,w,h) in nfaces:
        cv2.rectangle(frame, (x,y-50),(x+w,y+h+10), (0,250,0), 4 )
        ##get reagion of interest
        roi = gray_frame[y:y+h, x: x+w]
        ##since our actual images in our training file is of size 48,48 and grascale so we gotta preprocess accordingly
        cropped = np.expand_dims(np.expand_dims(cv2.resize(roi, (48,48)), -1), 0)

        ##now we pass thi cropped img to out model
        em_pred = model.predict(cropped)
        maxind = int(np.argmax(em_pred))
        cv2.putText(frame, label[maxind], (x+5,y-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('window-EMOTION_DETECTOR', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



