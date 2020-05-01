#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose, Concatenate
from tensorflow.keras import Model, regularizers

import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
from time import time
from tensorflow.keras.regularizers import l2


# In[2]:


path = "ML_forVidya/ML_steady_state_raw_data_folder_training/ml_raw_data_*.csv"
num_train = len(glob.glob(path))
power_map_train = np.zeros((num_train,34,32))
temp_map_train = np.zeros((num_train,34,32))
for im_num,fname in enumerate(glob.glob(path)):
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x = int(np.round(float(row[1])/2.5e-4))
            y = int(np.round(float(row[2])/2.5e-4))
            dyn_pow = float(row[3])
            leak_pow = float(row[4])
            alpha = float(row[5])
            power_map_train[im_num,x,y] = alpha*dyn_pow + leak_pow
            temp_map_train[im_num,x,y] = float(row[7])
max_temp = np.max(temp_map_train) 
max_power = np.max(power_map_train)
power_map_train = power_map_train/max_power
temp_map_train = temp_map_train/max_temp
power_map_train = power_map_train[...,np.newaxis]
temp_map_train = temp_map_train[...,np.newaxis]


# In[3]:


print(power_map_train.shape)
for im_num,power in enumerate(power_map_train):
    plt.figure()
    plt.imshow(np.squeeze(power))
    plt.figure()
    plt.imshow(np.squeeze(temp_map_train[im_num,...]))
    


# In[4]:


path = "ML_forVidya/ML_steady_state_raw_data_folder_testing/ml_raw_data_*.csv"
num_test = len(glob.glob(path))
power_map_test = np.zeros((num_test,34,32))
temp_map_test = np.zeros((num_test,34,32))
for im_num,fname in enumerate(glob.glob(path)):
    with open(fname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            x = int(np.round(float(row[1])/2.5e-4))
            y = int(np.round(float(row[2])/2.5e-4))
            dyn_pow = float(row[3])
            leak_pow = float(row[4])
            alpha = float(row[5])
            power_map_test[im_num,x,y] = alpha*dyn_pow + leak_pow
            temp_map_test[im_num,x,y] = float(row[7])
power_map_test = power_map_test/max_power
temp_map_test = temp_map_test/max_temp
power_map_test = power_map_test[...,np.newaxis]
temp_map_test = temp_map_test[...,np.newaxis]


# In[5]:


print(power_map_test.shape)
for im_num,power in enumerate(power_map_test):
    plt.figure()
    plt.imshow(np.squeeze(power))
    plt.figure()
    plt.imshow(np.squeeze(temp_map_test[im_num,...]))


# In[6]:


train_ds = tf.data.Dataset.from_tensor_slices(
    (power_map_train, temp_map_train)).batch(1)

test_ds = tf.data.Dataset.from_tensor_slices((power_map_test, temp_map_test)).batch(1)


# In[7]:


reg_rate =0.001
class encoder(Model):
  def __init__(self):
    super(encoder, self).__init__()
    self.conv1 = Conv2D(64, 3, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max1 = MaxPooling2D(2, padding='same')
    self.conv2 = Conv2D(32, 3, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max2 = MaxPooling2D(2, padding='same')
    self.conv3 = Conv2D(16, 5, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max3 = MaxPooling2D(2, padding='same')
#     self.dense = Dense(128,activation='relu',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    
  def call(self, x):
    x0 = self.conv1(x)
    
    x1 = self.max1(x0)
    x1 = self.conv2(x1)
    
    x2 = self.max2(x1)
    x2 = self.conv3(x2)
    
    x3 = self.max3(x2)
    
    return (x0,x1,x2,x3)

class decoder(Model):
  def __init__(self):
    super(decoder, self).__init__()
    self.conv0 = Conv2DTranspose(16, 7, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max1 = UpSampling2D(2)
    self.conv1 = Conv2DTranspose(32, 7, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max2 = UpSampling2D(2)
    self.conv2 = Conv2DTranspose(64, 3, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max3 = UpSampling2D(2)
    self.conv3 = Conv2DTranspose(1, 3, activation='relu',padding='SAME',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    


  def call(self, vals):
    x1 = self.conv0(vals[3])
    x1 = self.max1(x1) 
    x1_shape = tf.shape(vals[2])
    x1 = tf.slice(x1, tf.zeros(x1_shape.shape,dtype=tf.dtypes.int32), x1_shape)
    x1 = Concatenate()([x1, vals[2]])
    
    x2 = self.conv1(x1)
    x2 = self.max2(x2)
    x2_shape = tf.shape(vals[1])
#     print(x2_shape)
    x2 = tf.slice(x2, [0,0,0,0], x2_shape)
    x2 = Concatenate()([x2, vals[1]])
    
    x3 = self.conv2(x2)
    x3 = self.max3(x3)
    x3_shape = tf.shape(vals[0])
    x3 = tf.slice(x3, [0,0,0,0], x3_shape)
#     x3 = tf.slice(x3, tf.zeros(x3_shape.shape,dtype=tf.dtypes.int32), x3_shape)
    x3 = Concatenate()([x3, vals[0]])
    
    x4 = self.conv3(x3)
    return x4

class autoencoder(Model):
  def __init__(self):
    super(autoencoder, self).__init__()
    self.ae = encoder()
    self.de = decoder()

  def call(self, x):
    vals = self.ae(x)
    x = self.de(vals)
    return x


# Create an instance of the model
model = autoencoder()


# In[8]:


initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.98,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=['mse', 'mae', 'mape'])


# In[ ]:


st = time()
history = model.fit(train_ds, epochs=1500,
              #steps_per_epoch=195,
          validation_data=test_ds,
          validation_freq=1
          #validation_steps=3
          )
et = time()
tt = et-st
print("Elapsed time: %03d:%02d:%05.2f"%(int(tt/3600),int(tt/60)%60,tt%60))


# In[ ]:


from matplotlib import pyplot
# pyplot.plot(history.history['mse'])
# pyplot.plot(history.history['mae'])
pyplot.plot(history.history['mape'])
# pyplot.plot(history.history['cosine_proximity'])
pyplot.show()


# In[ ]:


y_pred = model.predict(test_ds)
print(y_pred.shape)
for im_num,temp in enumerate(y_pred):
    plt.figure()
    fig, axes = plt.subplots(2, 2)
    denorm_temp = np.squeeze(temp*max_temp)
    denorm_pred_temp  = (np.squeeze(temp_map_test[im_num,...])*max_temp)
    max_temp_im = max(np.max(denorm_temp),np.max(denorm_pred_temp))
    min_temp_im = min(np.min(denorm_temp),np.min(denorm_pred_temp))
    err = abs(denorm_pred_temp - denorm_temp)
    im = axes[0,1].imshow(denorm_pred_temp,vmin=0, vmax=max_temp_im)
    im = axes[1,0].imshow(err,vmin=0, vmax=max_temp_im)
    im = axes[0,0].imshow(denorm_temp,vmin=0, vmax=max_temp_im)
    axes[1,1].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
    print(np.max(err))


# In[ ]:


y_pred = model.predict(train_ds)
print(y_pred.shape)
for im_num,temp in enumerate(y_pred):
    plt.figure()
    fig, axes = plt.subplots(2, 2)
    denorm_temp = np.squeeze(temp*max_temp)
    denorm_pred_temp  = (np.squeeze(temp_map_train[im_num,...])*max_temp)
    max_temp_im = max(np.max(denorm_temp),np.max(denorm_pred_temp))
    min_temp_im = min(np.min(denorm_temp),np.min(denorm_pred_temp))
    err = abs(denorm_pred_temp - denorm_temp)
    im = axes[0,1].imshow(denorm_pred_temp,vmin=0, vmax=max_temp_im)
    im = axes[1,0].imshow(err,vmin=0, vmax=max_temp_im)
    im = axes[0,0].imshow(denorm_temp,vmin=0, vmax=max_temp_im)
    axes[1,1].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
    print(np.max(err))


# In[ ]:


path = "ML_forVidya/ml_modified.csv"
power_map = np.zeros((1,34,64))
temp_map = np.zeros((1,34,64))
with open(path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        x = int(np.round(float(row[1])/2.5e-4))
        y = int(np.round(float(row[2])/2.5e-4))
        dyn_pow = float(row[3])
        leak_pow = float(row[4])
        alpha = float(row[5])
        power_map[0,x,y] = alpha*dyn_pow + leak_pow
        temp_map[0,x,y] = float(row[7])
power_map = power_map/max_power
temp_map = temp_map/max_temp
power_map = power_map[...,np.newaxis]
temp_map = temp_map[...,np.newaxis]

y_pred = model.predict(power_map)
print(y_pred.shape)

plt.figure()
fig, axes = plt.subplots(2, 2)
denorm_temp = np.squeeze(y_pred*max_temp)
denorm_pred_temp  = (np.squeeze(temp_map)*max_temp)
max_temp_im = max(np.max(denorm_temp),np.max(denorm_pred_temp))
min_temp_im = min(np.min(denorm_temp),np.min(denorm_pred_temp))
err = abs(denorm_pred_temp - denorm_temp)
im = axes[0,1].imshow(denorm_pred_temp,vmin=0, vmax=max_temp_im)
im = axes[1,0].imshow(err,vmin=0, vmax=max_temp_im)
im = axes[0,0].imshow(denorm_temp,vmin=0, vmax=max_temp_im)
axes[1,1].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
print(np.max(err))


# In[ ]:




