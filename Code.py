# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 00:13:53 2017

@author: AMARESH
"""

#Age detection
% pylab inline
import os
import random

import pandas as pd
from scipy.misc import imread

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.index
test.index

data_dir= 'C:/Users/AMARESH/Documents/Machine Learning A-Z Template Folder/Age-Detection-NN'

#Randomly selecting an image
i = random.choice(train.index)
img_name = train.ID[i]
img = imread(os.path.join(data_dir, 'Train', img_name))
imshow(img)
print('Age: ', train.Class[i])
img.shape

#image resizing
from scipy.misc import imresize

temp=[]

for img_name in train.ID:
    img_path =  os.path.join(data_dir,'Train',img_name)
    img = imread(img_path)
    img = imresize(img,(32,32))
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

temp1 =[]
for img_name in test.ID:
    img_path =  os.path.join(data_dir,'Test',img_name)
    img = imread(img_path)
    img = imresize(img,(32,32))
    img = img.astype('float32')
    temp1.append(img)

test_x = np.stack(temp1)

#normalize the images
train_x = train_x/255
test_x = test_x/255

#Implementing Multi ANN

import keras

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
train_y = lb.fit_transform(train.Class)
train_y = keras.utils.np_utils.to_categorical(train_y)
train_y = train_y.astype('float32')

lb.classes_

'''from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
test_y = lb.fit_transform(test.Class)
test_y = keras.utils.np_utils.to_categorical(test_y)
test_y = test_y.astype('float32')'''

# Initialising the ANN
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer

input_num_units = (32, 32, 3)
hidden_num_units = 500
output_num_units = 3

epochs = 500
batch_size = 256

classifier = Sequential([
        InputLayer(input_shape= input_num_units),
        Flatten(),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =output_num_units, activation='softmax')
        ])

classifier.summary()

classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(train_x, train_y, batch_size = batch_size, epochs=epochs,verbose=1)

# Part 3 - Making the predictions and evaluating the model
classifier.fit(train_x, train_y, batch_size=batch_size,epochs=epochs,verbose=1, validation_split=0.2)

pred = classifier.predict_classes(test_x)
pred = lb.inverse_transform(pred)
lb.classes_
test['Class'] = pred
test.to_csv('sub08.csv', index=False)


# Initialising the CNN
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer,Convolution2D,MaxPooling2D

input_num_units = (32, 32, 3)
hidden_num_units = 500
output_num_units = 3

epochs = 5
batch_size = 256

classifier = Sequential([
        Convolution2D(50,(6,6),input_shape= input_num_units, activation='relu'),
        MaxPooling2D(pool_size = (3, 3)),
        Flatten(),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =hidden_num_units, activation='relu'),
        Dense(units =output_num_units, activation='sigmoid')
        ])

classifier.summary()
classifier.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(train_x, train_y, batch_size = batch_size, epochs=epochs,verbose=1)

c_pred = classifier.predict_classes(test_x)
c_pred = lb.inverse_transform(c_pred)

test['Class'] = c_pred
test.to_csv('sub05.csv', index=False)

#XGBoost
import xgboost
# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(train_x, train_y)

# Predicting the Test set results
y_pred = classifier.predict(test_x)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv = 10, verbose=1)
accuracies.mean()
accuracies.std()
