#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
from os import listdir
import numpy as np
import shutil
import random
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Accuracy
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# In[ ]:


test_path = 'C:/Users/Nakshata/Downloads/PlantVillage-20210617T081635Z-001/PlantVillage/test'
train_path = 'C:/Users/Nakshata/Downloads/PlantVillage-20210617T081635Z-001/PlantVillage/train'
valid_path = 'C:/Users/Nakshata/Downloads/PlantVillage-20210617T081635Z-001/PlantVillage/valid'


# In[ ]:


train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)


# In[17]:


mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()


# In[18]:


x = mobile.layers[-20].output
x = GlobalAveragePooling2D()(x)
output = Dense(units=13, activation='softmax')(x)
model = Model(inputs=mobile.input, outputs=output)
for layer in model.layers[:-23]:
    layer.trainable = False
model.summary()


# In[19]:


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=train_batches,
            steps_per_epoch=len(train_batches),
            validation_data=valid_batches,
            validation_steps=len(valid_batches),
            epochs=10,
            verbose=2)


# In[30]:


test_labels = test_batches.classes
predictions = model.predict(x=test_batches, steps=len(test_batches), verbose=0)


# In[36]:


cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
test_batches.class_indices
cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')


# In[ ]:




