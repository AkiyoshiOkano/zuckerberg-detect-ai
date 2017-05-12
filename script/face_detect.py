# -*- coding:utf-8 -*-

import cv2
import numpy as np

input_data_path = './gates_images/gates'
save_path = './cutted_gates_images/'
cascade_path = './opencv/data/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

image_count = 450
face_detect_count = 0

for i in range(image_count):
  img = cv2.imread(input_data_path + str(i) + '.jpg', cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)
  if len(face) > 0:
    for rect in face:
      # コメント外すと顔認識部分が赤線で囲まれる
      # cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0,255), thickness=1)
      # cv2.imwrite('detected.jpg', img)
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]
      cv2.imwrite(save_path + 'cutted_gates' + str(face_detect_count) + '.jpg', img[y:y+h, x:x+w])
      face_detect_count = face_detect_count + 1
  else:
    print 'image' + str(i) + ':NoFace'
