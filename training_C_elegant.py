# -*- coding: utf-8 -*-
"""
@author: Bipasha Kundu, Tanjemoon Ankur
Project 4
Pattern Recognition
"""

# -*- coding: utf-8 -*-

import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix


plt.close('all')
# Loading the Data
filelist = glob.glob('./Celegans_ModelGen/*/*.png')
print('Total images=  ' + str(len(filelist)))


X = np.array([np.array(cv2.resize(cv2.imread(fname,0),(50, 50), interpolation = cv2.INTER_CUBIC)) for fname in filelist])

print(X.shape)


y1 = np.zeros((5500, 1), dtype='int')
y2 = np.ones((5500, 1), dtype='int')
y = np.concatenate((y1,y2))

# Splitting into train and test
X, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,shuffle = True, random_state = 8)

# Visualizing Training Data:

for i in range(15):
    plt.subplot(5,5,i+1)
    plt.imshow(X[i], cmap=plt.cm.gray)
    plt.axis('off')
plt.show()

# reshaping and normalizing the data
X = X.reshape(X.shape[0], -1)/255
X_test = X_test.reshape(X_test.shape[0], -1)/255

# Training
t1 = time.time()



# defining the parameters for training
# Tried with  C=(0.1,1,10), kernel= ('rbf', 'poly') and gamma=(0.1)

model = svm.SVC(kernel= 'rbf', C=10, gamma=0.1)

# fit model 
model.fit(X, y_train.ravel())
t2 = time.time()-t1
print('Training Time= ' + str(round(t2,4)) + ' s')
t3 = time.time()

prediction = model.predict(X_test)
t4 = time.time()-t3
print('Testing Time = '  + str(round(t4,4)) + ' s')

# saving the trained model

filename = 'c_elegan_model.sav'
joblib.dump(model, filename)

# Plotting Confusion matrix and accuracy
accuracy = ((prediction.ravel() == y_test.ravel())*1).sum()/len(X_test)*100
conf_matrix = confusion_matrix(y_test, prediction)

plot_confusion_matrix(model, X_test, y_test)  
plt.title('Test Accuracy = ' + str(round(accuracy, 2)),' %')
plt.show()

