# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
import os
import random
import cv2
from sklearn.metrics import confusion_matrix

model = load_model('my_model.h5')

fileNames = []

fileList = os.listdir("./dataset/Test/Normal/")
for fileName in fileList:
    fileNames.append('./dataset/test/Normal/'+fileName)
fileList = os.listdir("./dataset/Test/C/")
for fileName in fileList:
    fileNames.append('./dataset/test/C/'+fileName)

random.shuffle(fileNames)  
yTrue = [0 if i.find("./dataset/test/Normal/n") else 1 for i in fileNames]

images = []
for i in fileNames:
    img = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    img = np.resize(img, (128, 128, 1))
    images.append(img)

yPred = model.predict(np.array(images), batch_size=1, verbose=1)
yPred.flatten()
yPred = np.rint(yPred)
print(confusion_matrix(yTrue, yPred))
