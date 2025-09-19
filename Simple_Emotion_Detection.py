import cv2
import numpy as np
face_cascade= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap= cv2.VideoCapture(0)
print("Simple Emotion Detection - Press 'q' to quit")
while True:
    ret, frame= cap.read()
    if not ret:
        break
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        face_roi= gray[y:y+h, x:x+w]
        face_area= w*h
        if face_area> 10000:
            emotion_text= "Happy Face Detected"
            color= (0,255,0)
        elif face_area>5000:
            emotion_text= "Natural Face Detected"
            color= (255, 255, 0)
        else:
            emotion_text= "Face Detected"
            color= (255, 0, 255)
        cv2.putText(frame, emotion_text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if len(faces)==0:
            cv2.putText(frame, "No Face Detected",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Simple Emotion Detection",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. GoodBye!")