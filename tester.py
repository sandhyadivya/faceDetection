import cv2
import os
import numpy as np
import faceDetection as fr
test_img=cv2.imread('C:/python/virat kohli.jpg')
faces_detected,gray_img=fr.face_detection(test_img)
print("faces_detected:",faces_detected)
for (x,y,w,h) in faces_detected:
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(250,0,0),thickness=5)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face.detection.tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows
