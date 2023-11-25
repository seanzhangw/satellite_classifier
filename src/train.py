#!/usr/bin/env python

import scipy.io 
from sklearn import svm
import numpy as np
import colorsys
import pickle
mat = scipy.io.loadmat('./sat-4-full.mat')

train_x = mat['train_x']
train_y = mat['train_y']
test_x = mat['test_x']
test_y = mat['test_y']
annotations = mat['annotations']

# indexes through each image, adds each pixel rgb value to list rgbvalues as a 
# list, [[red],[green],[blue]]. This list is converted to hsv and the mean & std 
# values of hsv are calculated for each image.
def calculatefeatures(train_x, train_y):
    labels = []
    hsvmeans = []
    hsvstd = []
    data = []
    accum = 0
    train_x = train_x.transpose((3, 0, 1, 2))

    for row in range(len(train_y[0])):
        for col in range(len(train_y)):
            if train_y[col][row] == 1: 
                labels.append(col)
                if col == 0:
                    data.append("barren land")
                if col == 1:
                    data.append("trees")    
                if col == 2:
                    data.append("grassland")  
                if col == 3:
                    data.append("none")  

    for index in train_x:
        rgbvalues = []
        hsvvalues = []
        for row in range(len(index)):
            for col in range(len(index[0])):
                rgbvalues.append([index[row][col][0], index[row][col][1], index[row][col][2]])
                hsvvalues.append(colorsys.rgb_to_hsv(rgbvalues[row*col][0]/255, rgbvalues[row*col][1]/255, rgbvalues[row*col][2]/255))

        hsvmeans.append(np.mean(hsvvalues, axis=0))
        hsvstd.append(np.std(hsvvalues, axis=0))

        accum += 1
        if accum %100000== 0:
            print(accum)
    
    return np.hstack((hsvmeans, hsvstd)), labels, data


features, labels, data = calculatefeatures(train_x, train_y)
rbfmodel = svm.SVC(C=2511, gamma=10, kernel='rbf')
rbfmodel.fit(features,labels)

with open('model.dat', 'wb') as f:
    pickle.dump(rbfmodel, f)