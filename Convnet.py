# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 18:06:28 2020

@author: Heisenberg
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import cv2
import matplotlib.pyplot as plt
import os
import seaborn as sns
from PIL import Image
from scipy import misc
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import misc
from random import shuffle
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.initializers import glorot_uniform
from keras.models import Model, load_model
os.chdir('C:\\Users\\Heisenberg\\Desktop\\Certificates\\Age detection actors classification\\train_DETg9GD\\Train\\')
df=pd.read_csv('train2.csv')
young=[]
middle=[]
old=[]
for i in range(len(df)):
    if(df['Class'][i]=='YOUNG'):
        young.append(df['ID'][i])
    elif(df['Class'][i]=='MIDDLE'):
        middle.append(df['ID'][i])
    else:
        old.append(df['ID'][i])
onlyfiles = os.listdir()
onlyfiles.pop()
shuffle(onlyfiles)

X_data =[]
for file in onlyfiles:
    face = misc.imread(file)
    face =cv2.resize(face, (32, 32) )
    X_data.append(face)

X = np.squeeze(X_data)
X.shape
X = X.astype('float32')
X /= 255

classes=onlyfiles

for i in range(len(classes)):
    for j in range(len(middle)):
        if(classes[i]==middle[j]):
            classes[i]=1
    for j in range(len(young)):
        if(classes[i]==young[j]):
            classes[i]=0
    for j in range(len(old)):
        if(classes[i]==old[j]):
            classes[i]=2


categorical_labels = to_categorical(classes, num_classes=3)

categorical_labels[:10]

(x_train, y_train), (x_test, y_test) = (X[:15000],categorical_labels[:15000]) , (X[15000:] , categorical_labels[15000:])
(x_valid , y_valid) = (x_test[:1900], y_test[:1900])
(x_test, y_test) = (x_test[1900:], y_test[1900:])

len(x_train)+len(x_test) + len(x_valid) == len(X)


model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(32,32,3))) 
model.add(tf.keras.layers.BatchNormalization(axis=1))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.BatchNormalization(axis=1))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
#model.add(tf.keras.layers.BatchNormalization(axis=1))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.5))


model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.BatchNormalization(axis=1))


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Take a look at the model summary
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=50,
         validation_data=(x_valid, y_valid))



# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

labels=["YOUNG","MIDDLE","OLD"]
#Test set
os.chdir('C:\\Users\\Heisenberg\\Desktop\\Certificates\\Age detection actors classification\\test_Bh8pGW3\\Test\\')
onlyfiles2 = os.listdir()
shuffle(onlyfiles2)

X_test_data =[]
for file in onlyfiles2:
    face = misc.imread(file)
    face =cv2.resize(face, (32, 32) )
    X_test_data.append(face)

X_test = np.squeeze(X_test_data)
X_test.shape
X_test = X_test.astype('float64')
X_test /= 255

y_hat = model.predict(X_test)
y_hat=pd.DataFrame(y_hat)
y_hat.rename(columns={0: "YOUNG", 1: "MIDDLE",2: "OLD"},inplace=True)
out=y_hat.idxmax(axis=1)
out=pd.DataFrame(out)
onlyfiles2=pd.DataFrame(onlyfiles2)

df=pd.concat([out,onlyfiles2],axis=1)
output=df.to_csv('Test.csv')
