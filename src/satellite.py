#!/usr/bin/env python
import scipy.io 
from sklearn import svm
import numpy as np
import colorsys
import csv
import matplotlib.pyplot as plt
import pickle

mat = scipy.io.loadmat('/classes/ece2720/fpp/test_x_only.mat')

# train_x = mat['train_x']
# train_y = mat['train_y']
test_x = mat['test_x']
# test_y = mat['test_y']
# annotations = mat['annotations']

# indexes through each image, adds each pixel rgb value to list rgbvalues as a 
# list, [[red],[green],[blue]]. This list is converted to hsv and the mean & std 
# values of hsv are calculated for each image.
def calculatefeatures(train_x):
    labels = []
    hsvmeans = []
    hsvstd = []
    accum = 0
    train_x = train_x.transpose((3, 0, 1, 2))

    # for row in range(len(train_y[0])):
    #     for col in range(len(train_y)):
    #         if train_y[col][row] == 1: 
    #             labels.append(col) 

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
    
    return np.hstack((hsvmeans, hsvstd)), labels

# Define the start and end powers (base 10)
# linearlist = []
# rbflist=[]
# polylist=[]
# xlist = []
# Loop over powers of 10 from start_power to end_power to find optimal C 
# for power in np.arange(-5, 6,0.4):
#     C = 10 ** power
#     features, labels, data = calculatefeatures(train_x, train_y)
#     # linearmodel = svm.SVC(C=C, kernel='linear')
#     # linearmodel.fit(features, labels[:1000])

#     rbfmodel = svm.SVC(C=C, kernel='rbf')
#     rbfmodel.fit(features,labels[:1000])

#     # polymodel = svm.SVC(C=C, kernel='poly')
#     # polymodel.fit(features,labels[:1000])

#     features, labels, data = calculatefeatures(test_x, test_y)
#     # linearlist.append(linearmodel.score(features,labels[:1000]))
#     # rbflist.append(rbfmodel.score(features,labels[:1000]))
#     # polylist.append(polymodel.score(features,labels[:1000]))
#     xlist.append(C)

# plt.plot(xlist, linearlist, label="Linear")
# plt.plot(xlist,rbflist, label="Radial Basis Function")
# plt.plot(xlist,polylist,label="Polynomial")
# plt.legend(loc="lower right")
# plt.xlabel("C hyperparameter value")
# plt.xscale('log')
# plt.ylabel("Mean accuracy of test data set")
# plt.show()

# for gammaval in np.arange(-3, 6,0.4):
#     print(gammaval)
#     g = 10 **gammaval
#     features, labels, data = calculatefeatures(train_x, train_y)
#     rbfmodel = svm.SVC(C=2511, gamma=g, kernel='rbf')
#     rbfmodel.fit(features,labels[:1000])
    
#     features, labels, data = calculatefeatures(test_x, test_y)
#     rbflist.append(rbfmodel.score(features,labels[:1000]))
#     xlist.append(g)

# plt.plot(xlist, rbflist)
# plt.xscale("log")
# plt.xlabel("Gamma hyperparameter value")
# plt.ylabel("Mean accuracy of test data set")
# plt.show()

# features, labels, data = calculatefeatures(train_x, train_y)
# rbfmodel = svm.SVC(C=2511, gamma=10, kernel='rbf')
# rbfmodel.fit(features,labels)

# with open('model.dat', 'wb') as f:
#     pickle.dump(rbfmodel, f)
with open('model.dat', 'rb') as f:
    model = pickle.load(f)

features, labels = calculatefeatures(test_x)
print(model.score(features,labels))
predictions = model.predict(features)

data = []
for col in predictions:
    if col == 0:
        data.append("barren land")
    if col == 1:
        data.append("trees")    
    if col == 2:
        data.append("grassland")  
    if col == 3:
        data.append("none") 

with open("landuse.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(data)
