#!/usr/bin/env python
# coding: utf-8
from sklearn import datasets
import numpy as np
from sklearn.svm import SVC

#Imported data from UCI Machine Learning Repository
iris = datasets.load_iris()
x = iris.data
y = iris.target


#Error Check
assert np.mean(x)>0
assert np.mean(y)>0

#Filtered out certain flower labels
#only looking at charachterisitc 0 and 2
#excludes label 0 -> only label 1 and 2
zipped=np.column_stack((x, y))
zipped=list(map(lambda zipped: [zipped[0], zipped[2], zipped[-1]], zipped))
zipped=[entry for entry in zipped if entry[-1]!=0.0]
zipped=np.asarray(zipped)
data_set=zipped[:,[0,1]]
labels=zipped[:,[2]]

#Printing iris target data
print(y)



import matplotlib.pyplot as plt
#Getting the length to make the array
l = len([val for val in labels if val ==1])

dataArray = data_set[:l]
redFlower1 = dataArray.T[0]
redFlower2 = dataArray.T[1]

dataArray2 = data_set[l:]
blueFlower1 = dataArray2.T[0]
blueFlower2 = dataArray2.T[1]

#Creating the scatter plot
plt.scatter(redFlower1, redFlower2, color="red")
plt.scatter(blueFlower1, blueFlower2, color="blue")
plt.xlabel("length (cm)")
plt.ylabel("width (cm)")
plt.grid()
plt.show()

#
print(np.subtract([2,3,5], 4.0))



import math
#Machine Learning utilizing Linear Support Vectors
#Scikit-learn: Machine Learning in Python,
#Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
d=dict()
print('  C \t | training error     |     number of support vectors')
for i in range(-3,6,1):

    C=math.pow(10,i)
    clf = SVC(C,kernel='linear')
    clf.fit(data_set,labels.ravel())
    error=1-clf.score(data_set, labels.ravel())

    d[C]=(round(error,4),len(clf.support_vectors_))
    print('{}  \t, \t{}\t\t, \t{}'.format(C, d[C][0], d[C][1]))


minimum_C=min(d, key=d.get)
print("The Value of C with the minimum test error and minimum number of support vectors is:", minimum_C)
clf = SVC(minimum_C,kernel='linear')
clf.fit(data_set,labels.ravel())


w=clf.coef_[0]
intercept=clf.intercept_[0]




import matplotlib.pyplot as plt
#from lplot import abline
#Machine Learning utilizing Slope-Intercept form

#Function that plots a line given a slope and intercept
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

#Getting the length to make the array, again
l = len([val for val in labels if val ==1])

dataArray = data_set[:l]
redFlower1 = dataArray.T[0]
redFlower2 = dataArray.T[1]

dataArray2 = data_set[l:]
blueFlower1 = dataArray2.T[0]
blueFlower2 = dataArray2.T[1]

xdata = np.array(range(0, 10))
ydata=(-w[0]/w[1])*xdata-intercept/w[1]
plt.plot(xdata, ydata, color="blue")

##lp
#abline((-w[0]/w[1]),(intercept/w[1]))

#Creating the scatter plot
plt.scatter(redFlower1, redFlower2, color="red")
plt.scatter(blueFlower1, blueFlower2, color="blue")
plt.grid()
plt.show()
