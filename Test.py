# -*- coding: utf-8 -*-
"""
@author: Bipasha Kundu, Tanjemoon Ankur
Project 4
Pattern Recognition
"""
import cv2 
import glob
import joblib
import numpy as np
import tkinter as tk 
from tkinter.filedialog import askdirectory

root = tk.Tk()
root.withdraw()

"""Name of the directory path containing test images"""
path = askdirectory()
print('User Sleected Directory ' + path)

filelist = glob.glob(path + '/*.png')
X = np.array([np.array(cv2.resize(cv2.imread(fname,0),(50, 50), interpolation = cv2.INTER_CUBIC)) for fname in filelist])

""" Loading the Trained Model """
model=joblib.load('c_elegan_model.sav')

""" Testing if there is a worm or not  """
if X.shape[0] > 0:
    X = X.reshape(X.shape[0], -1)/255
    predicted_c_elegans = model.predict(X).ravel()

""" Making Tallies with the corresponding label : 1 (worm) or 0 (no worm)"""

if X.shape[0] > 0:
    for i, row in enumerate(filelist):
        print(row.split(sep='/')[-1] + ' is an image of ' + str(predicted_c_elegans[i]))
        if predicted_c_elegans[i]==1.0:
            print(' Detected C elegan is of worm')
        if predicted_c_elegans[i]==0.0:
            print('Detected C elegan is of No worm')

""" Total Number of Images with worm and no_worm"""

worm=sum(predicted_c_elegans==1)
no_worm=len(filelist)-worm
print('No of Worm=', worm)
print('No of No_worm=',no_worm)
    

