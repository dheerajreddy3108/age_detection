# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 09:31:40 2020

@author: dheeraj_reddy_pera,
"""
#Importing reqquired libraries for performing the gender classification task

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import math
from keras.models import Sequential
from keras.layers import Flatten, Conv2D,MaxPooling2D,BatchNormalization,Dropout,Dense
from keras.utils.np_utils import to_categorical
from PIL import Image
from sklearn.model_selection import train_test_split
#specifying the path of the files
path = os.chdir('/ML/utkface')
current_path=os.chdir('UTKFace')

#reading a image
img= Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((1,128,128))
#Specifying only to consider files in specified path
files = os.listdir()
files_count=len(files)
#extracting ages
age = [i.split('_')[0] for i in files]
#creating classes based on the age
classes =[]
#splitting the classes according to the age
"if age is between 0-12:children ;12-25:adult;25-40:young age:40-60:middle age; older than 60:old" 
                    
for i in age:
    i=int(i)
    if i<=12:
        classes.append(0)
    if(i>12 and i<=25):
        classes.append(1)
    if(i>25 and i<40):
        classes.append(2)
    if(i>=40 and i<60):
        classes.append(3)
    if(i>60):
        classes.append(4)

#image to vector
X = []
for file in files:
    faces = cv2.imread(file)
    faces=cv2.resize(faces,(32,32))
    X.append(faces)
    
X=np.squeeze(X)
#normalizing the data
X= X.astype('float32')/255
#changing classes
classes[:10]
labels=to_categorical(classes,num_classes=5)
labels[:10]

#Data spltiing
X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.2,random_state=123,shuffle=True)

#building up the model
model= Sequential()
model.add(Conv2D(128,kernel_size=(3,3),input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(65,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(5,activation='softmax'))

model.summary()

#model Compiling
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#model fit
model.fit(X_train,y_train,batch_size=64,epochs=32,validation_data=(X_test,y_test))

#model evaluation metrics
test_score=model.evaluate(X_test,y_test)

label=['child','youth','adult','middle age','old']
#model prediction
y_predict=model.predict(X_test)


figure=plt.figure(figsize=(20,10))
for j, index in enumerate(np.random.choice(X_test.shape[0],size=15,replace=False)):
    a= figure.add_subplot(3,5,i+1,x_ticks=[], y_ticks=[])
    a.imshow(np.squeeze(X_test[index]))
    pred_index=np.argmax(y_predict[index])
    true_index=np.argmax(y_test[index])
    a.set_title("{}"('{'}).format(label[pred_index],label[true_index]),color=('blue' if pred_index == true_index else 'red'))
plt.show()

    