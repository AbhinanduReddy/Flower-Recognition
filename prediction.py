#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import cv2
import pandas as pd
import time


# In[2]:


df=pd.read_csv(r'C:\Users\Abhinandu Reddy\Desktop\wpi cpi\data\trainflowers.csv')


# In[3]:


df.head()


# In[4]:


df['category'][0]


# In[ ]:


import numpy as np
a=time.time()
train=[]
for i in range(0,18540):
    path=os.path.join(r'C:\Users\Abhinandu Reddy\Desktop\wpi cpi\data\train',str(i)+'.jpg')
    if i%500==0:
        print(i,end=' ')
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(60,60))

    train.append([img])
b=time.time()
print((b-a)*100*10000)


# In[138]:


import numpy as np
a=time.time()
test=[]

for i in range(18540,20549):
    path=os.path.join(r'C:\Users\Abhinandu Reddy\Desktop\wpi cpi\data\test',str(i)+'.jpg')
    if i%500==0:
        print(i,end=' ')
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img,(60,60))

    test.append(img)
b=time.time()
print((b-a)*100*10000)


# In[162]:


x=np.array(train)


# In[168]:


x=np.squeeze(x)
x=np.expand_dims(x,axis=3)
x.shape


# In[185]:


train.head()
train.to_csv('flowers')


# In[7]:


y=train[1]
t=train[0]


# In[147]:


y=df['category']


# In[8]:


q=[]
for i in range(18540):
    q.append(train[0][0])
q=np.asanyarray(q)
q=q.reshape((18540,60,60,1))
q[1:].shape


# In[111]:


cv2.imshow('d',train[0])
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[144]:


from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model


# In[170]:


xinput=Input(q.shape[1:])
x=Conv2D(32,(7,7),strides=(1,1),name='conv0')(xinput)
x=Activation('relu')(x)
x=MaxPooling2D((2,2),name='max_pool')(x)
x=Flatten()(x)
x=Dense(64,activation='relu',name='f')(x)
x=Dense(103,activation='softmax',name='f1')(x)

model=Model(inputs=xinput,outputs=x,name='model')


# In[171]:


model.summary()


# In[172]:


model.compile('adam','sparse_categorical_crossentropy',metrics=['accuracy'])


# In[173]:


np.unique(y)


# In[174]:


model.fit(q,y,epochs=2)


# In[131]:


x.shape


# In[249]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=60,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')


# In[262]:


import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint_name = 'Weights-{epoch:04d}--{loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(60, 60, 1)),
    tf.keras.layers.AveragePooling2D(3,3),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.AveragePooling2D(3, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.tanh),
    tf.keras.layers.Dense(103, activation=tf.nn.softmax)])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_datagen.flow(x,y, batch_size=32),
                             steps_per_epoch=len(q) / 32,
                              epochs=80,callbacks=callbacks_list)
#model.fit(q,y,batch_size=40,epochs=10)


# In[261]:


model.summary()


# In[181]:


xt=np.array(test)


# In[182]:


xt.shape


# In[183]:


xt=np.expand_dims(xt,axis=3)


# In[184]:


xt.shape


# In[185]:


p=model.predict(xt)


# In[219]:


pp=[]


# In[220]:


for i in range(2009):
    pp.append(np.argmax(p[i]))


# In[221]:


pp=np.array(pp)


# In[222]:


pp.shape


# In[228]:


samfl=pd.read_csv(r'C:\Users\Abhinandu Reddy\Desktop\wpi cpi\data\sample_submissionflowers.csv')


# In[232]:


samfl['category']=pp


# In[237]:


samfl.to_csv('samfl')


# In[ ]:




