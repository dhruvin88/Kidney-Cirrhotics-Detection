# -*- coding: utf-8 -*-
import glob
import os
import random
import numpy as np
import cv2

dataDir = './Output_Images/'

listOfFiles = glob.glob(dataDir+"*.pgm")
random.shuffle(listOfFiles)
random.shuffle(listOfFiles)
    
train_X, val_X, test_X =np.split(listOfFiles, [int(.6 * len(listOfFiles)), \
                                               int(.8 * len(listOfFiles))])    
    
# 1 if normal  0 if cirrhotics
train_Y = [1 if i.find(dataDir+"n") else 0 for i in train_X]
val_Y = [1 if i.find(dataDir+"n") else 0 for i in val_X]
test_Y = [1 if i.find(dataDir+"n") else 0 for i in test_X]
    
#empty of previous training, testing, and validation data
fileList = os.listdir("./dataset/training/Normal/")
for fileName in fileList:
    os.remove('./dataset/training/Normal/'+"/"+fileName)
        
fileList = os.listdir("./dataset/training/c/")
for fileName in fileList:
    os.remove('./dataset/training/c/'+"/"+fileName)
    
#test
fileList = os.listdir("./dataset/test/Normal/")
for fileName in fileList:
    os.remove('./dataset/test/Normal/'+"/"+fileName)
        
fileList = os.listdir("./dataset/test/c/")
for fileName in fileList:
    os.remove('./dataset/test/c/'+"/"+fileName)
        
#validation
fileList = os.listdir("./dataset/Validation/Normal/")
for fileName in fileList:
    os.remove('./dataset/Validation/Normal/'+"/"+fileName)
        
fileList = os.listdir("./dataset/Validation/c/")
for fileName in fileList:
    os.remove('./dataset/Validation/c/'+"/"+fileName)
    
#set training and test data
for x in range(len(train_X)):
    if train_Y[x] == 0:
        img = cv2.imread(train_X[x],0)
        fileName = os.path.basename(train_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/training/Normal/"+fileName+".jpg", img)
    else:
        img = cv2.imread(train_X[x],0)
        fileName = os.path.basename(train_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/training/C/"+fileName+".jpg", img)
for x in range(len(test_X)):
    if test_Y[x] == 0:
        img = cv2.imread(test_X[x],0)
        fileName = os.path.basename(test_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/test/Normal/"+fileName+".jpg", img)
    else:
        img = cv2.imread(test_X[x],0)
        fileName = os.path.basename(test_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/test/C/"+fileName+".jpg", img)
    
for x in range(len(val_X)):
    if val_Y[x] == 0:
        img = cv2.imread(test_X[x],0)
        fileName = os.path.basename(test_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/Validation/Normal/"+fileName+".jpg", img)
    else:
        img = cv2.imread(test_X[x],0)
        fileName = os.path.basename(test_X[x])
        fileName,_ = os.path.splitext(fileName)
        cv2.imwrite("./dataset/Validation/C/"+fileName+".jpg", img)

