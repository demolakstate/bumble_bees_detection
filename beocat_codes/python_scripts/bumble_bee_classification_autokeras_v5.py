#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # Image classification

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/images/classification"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/images/classification.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This tutorial shows how to classify images of flowers. It creates an image classifier using a `keras.Sequential` model, and loads data using `preprocessing.image_dataset_from_directory`. You will gain practical experience with the following concepts:
# 
# * Efficiently loading a dataset off disk.
# * Identifying overfitting and applying techniques to mitigate it, including data augmentation and Dropout.
# 
# This tutorial follows a basic machine learning workflow:
# 
# 1. Examine and understand data
# 2. Build an input pipeline
# 3. Build the model
# 4. Train the model
# 5. Test the model
# 6. Improve the model and repeat the process

# In[ ]:





# ## Import TensorFlow and other libraries

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[3]:


#import glob


# ## Download and explore the dataset

# In[4]:


data_dir = 'dataset_2/training'


# In[ ]:





# # Load using keras.preprocessing
# 
# Let's load these images off disk using the helpful [image_dataset_from_directory](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory) utility. This will take you from a directory of images on disk to a `tf.data.Dataset` in just a couple lines of code. If you like, you can also write your own data loading code from scratch by visiting the [load images](https://www.tensorflow.org/tutorials/load_data/images) tutorial.

# ## Create a dataset

# Define some parameters for the loader:

# In[5]:


batch_size = 16
img_height = 180
img_width = 180


# It's good practice to use a validation split when developing your model. Let's use 80% of the images for training, and 20% for validation.

# In[6]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[7]:


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# You can find the class names in the `class_names` attribute on these datasets. These correspond to the directory names in alphabetical order.

# In[8]:


class_names = train_ds.class_names
print(class_names)


# ## Visualize the data
# 
# Here are the first 9 images from the training dataset.

# In[9]:


#import matplotlib.pyplot as plt

#plt.figure(figsize=(10, 10))
#for images, labels in train_ds.take(1):
#  for i in range(9):
#    ax = plt.subplot(3, 3, i + 1)
#    plt.imshow(images[i].numpy().astype("uint8"))
#    plt.title(class_names[labels[i]])
#    plt.axis("off")


# You will train a model using these datasets by passing them to `model.fit` in a moment. If you like, you can also manually iterate over the dataset and retrieve batches of images:

# In[10]:


#for image_batch, labels_batch in train_ds:
#  print(image_batch.shape)
#  print(labels_batch.shape)
#  break


# The `image_batch` is a tensor of the shape `(32, 180, 180, 3)`. This is a batch of 32 images of shape `180x180x3` (the last dimension refers to color channels RGB). The `label_batch` is a tensor of the shape `(32,)`, these are corresponding labels to the 32 images. 
# 
# You can call `.numpy()` on the `image_batch` and `labels_batch` tensors to convert them to a `numpy.ndarray`.
# 

# ## Configure the dataset for performance
# 
# Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data.
# 
# `Dataset.cache()` keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# 
# `Dataset.prefetch()` overlaps data preprocessing and model execution while training. 
# 
# Interested readers can learn more about both methods, as well as how to cache data to disk in the [data performance guide](https://www.tensorflow.org/guide/data_performance#prefetching).

# In[11]:


#AUTOTUNE = tf.data.AUTOTUNE

#train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[12]:


#get_ipython().system('pip install autokeras')


# In[13]:


from autokeras import ImageClassifier


# In[ ]:


# Initialize the image classifier.
model = ImageClassifier(max_trials=1)
history = model.fit(train_ds, validation_data=val_ds, verbose=True, epoch=1)


# In[ ]:





# ## Visualize training results

# Create plots of loss and accuracy on the training and validation sets.

# In[ ]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('acc_loss_plot.png')
plt.show()


# In[ ]:


## Evaluation of model accuracy on validation data ##

### Let's compare how the model performs on the validation dataset ###

test_loss, test_acc = model.evaluate_generator(val_ds, verbose=0)

print('\nValidation accuracy:', test_acc)

print('\nValidation loss:', test_loss)


# In[ ]:




## Make predictions on test data ##

### Let's make predictions on some images ###

val_images_list = []
val_labels_list = []
for val_images, val_labels in val_ds: # get batches
  for val_image, val_label in zip(val_images,val_labels):
    val_images_list.append(val_image)
    val_labels_list.append(val_label)


len(val_images_list)

val_images_list = np.array(val_images_list)
val_labels_list = np.array(val_labels_list)

predictions = model.predict(val_images_list)

predictions[0]

np.argmax(predictions[0])

predictions[-1]

np.argmax(predictions[-1])


# In[ ]:



## Confusion Matrix ##

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report

y_true = val_labels_list
y_pred = np.array([np.argmax(x) for x in predictions])

y_pred

cm = confusion_matrix(y_true, y_pred)

print(cm)

plt.matshow(cm, cmap=plt.cm.gray)
plt.show()

figure = plt.figure()
axes = figure.add_subplot(111)

caxes = axes.matshow(cm)
figure.colorbar(caxes)

axes.set_xticklabels(['']+class_names)
axes.set_yticklabels(['']+class_names)

plt.show()


# In[ ]:



## Plot on Errors ##

row_sums = cm.sum(axis=1, keepdims=True)
norm_cm = cm / row_sums



np.fill_diagonal(norm_cm, 0)
plt.matshow(norm_cm, cmap=plt.cm.gray)
plt.show()

## Confusion Matrix Heat Map ##

#!pip install seaborn


# In[ ]:


import seaborn as sb

heat_map = sb.heatmap(cm, annot=True)
sb.set(font_scale=1)
plt.show()

for index, specie in enumerate(class_names):
  print((index, specie))

## Classification report of model on test set ##

print(classification_report(y_true, y_pred, target_names=class_names))
print('Legend')
for index, specie in enumerate(class_names):
  print((index, specie))

y_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


