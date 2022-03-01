#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Flatten, Dropout,BatchNormalization, GlobalAveragePooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD 
import pandas as pd
import cv2 as cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os 
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[2]:


dataset = r"/home/sachin269/Downloads/ChestXRay/chest_xray/train"
Normal_path = r"/home/sachin269/Downloads/ChestXRay/chest_xray/train/NORMAL"
Pneumonia_path = r"/home/sachin269/Downloads/ChestXRay/chest_xray/train/PNEUMONIA/"


# In[3]:


img = cv2.imread(Normal_path+'/IM-0115-0001.jpeg')
print(img.shape)
plt.imshow(img)


# In[4]:


vals = [Normal_path, Pneumonia_path]
print(os.listdir(vals[0]).__len__())
print(os.listdir(vals[1]).__len__())


# In[6]:


pathdir = [Normal_path, Pneumonia_path]
classes = ['Normal', 'Pneumonia']
filepaths = []
labels = []
for i, j in zip(pathdir, classes):
    filelist = os.listdir(i)
#     print(filelist)
    for vals in filelist:
        x = os.path.join(i, vals)
        filepaths.append(x)
        labels.append(j)
# print(filepaths.__len__(), labels.__len__())


# In[7]:


print(filepaths[0:4])
print(labels[0:4])

print(filepaths[-4:])
print(labels[-4:])


# In[8]:


dataset = list(zip(filepaths, labels))
pathframe = pd.DataFrame(dataset, columns=['filepaths', 'labels'])


# In[9]:


pathframe.__len__()
pathframe.tail()


# In[10]:


print(pathframe['labels'].value_counts())


# In[11]:


for i in range(0, 20):
    vals = np.random.randint(1, len(pathframe))
    plt.subplot(4,5, i+1)
    plt.imshow(cv2.imread(pathframe.filepaths[vals]))
    plt.axis('off')
plt.show()


# In[12]:


Train, Test = train_test_split(pathframe, train_size=0.90, random_state=0)
Train_new, valid = train_test_split(Train, train_size = 0.90, random_state=0)
print(Train.shape, Test.shape, Train_new.shape, valid.shape)


# In[13]:


train_datagen = ImageDataGenerator(rescale=1.0/255, rotation_range= 40 , width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
                                  zoom_range=0.2, horizontal_flip = True, vertical_flip= True)
test_datagen = ImageDataGenerator(rescale=1.0/255)


# In[14]:


train_gen = train_datagen.flow_from_dataframe(dataframe = Train_new, x_col = 'filepaths', y_col='labels', batch_size=16, 
                                             target_size=(250,250), class_mode = 'binary', shuffle=True)

valid_gen = train_datagen.flow_from_dataframe(dataframe = valid, x_col = 'filepaths', y_col='labels', batch_size=16, 
                                             target_size=(250,250), class_mode = 'binary', shuffle=True)
test_gen = train_datagen.flow_from_dataframe(dataframe = Test, x_col = 'filepaths', y_col='labels', batch_size=16, 
                                             target_size=(250,250), class_mode = 'binary', shuffle=False)


# In[15]:


print(train_gen.class_indices)
print(train_gen[0][0].shape)

for i in range(0, 12):
    val = train_gen[0][0][i]
    vals = val.astype('uint8')
    plt.subplot(4,3,i+1)
    plt.imshow(vals)
    plt.axis('off')
plt.show()


# In[16]:


model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape = (250, 250, 3), activation = 'relu'))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 1, activation = 'sigmoid'))

callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience = 2, min_delta= 0.01)

optim=tf.keras.optimizers.RMSprop(learning_rate=0.01, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer = optim, loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model.fit(train_gen, validation_data= valid_gen, epochs=5)

model.summary()


# In[17]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
plt.show()


# In[18]:


model.evaluate(test_gen)


# In[19]:


import seaborn as sns
import matplotlib.pyplot as plt
#Violin Plots for all the weights matrices.
w_after = model.get_weights()

h1_w = w_after[0].flatten().reshape(-1,1)
h2_w = w_after[2].flatten().reshape(-1,1)
h3_w = w_after[4].flatten().reshape(-1,1)
out_w = w_after[6].flatten().reshape(-1,1)


fig = plt.figure(figsize=(12,10))
plt.title("Weight matrices after model is trained")
plt.subplot(1, 4, 1)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h1_w,color='b')
plt.xlabel('Hidden Layer 1')

plt.subplot(1, 4, 2)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h2_w, color='r')
plt.xlabel('Hidden Layer 2 ')

plt.subplot(1, 4, 3)
plt.title("Trained model Weights")
ax = sns.violinplot(y=h3_w, color='g')
plt.xlabel('Hidden Layer 3 ')


# In[ ]:




