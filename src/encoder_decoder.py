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


class encoder(Model):
  def __init__(self):
    super(encoder, self).__init__()
    self.conv1 = Conv2D(64, 3, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max1 = MaxPooling2D(2, padding='same')
    self.conv2 = Conv2D(32, 3, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max2 = MaxPooling2D(2, padding='same')
    self.conv3 = Conv2D(16, 5, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max3 = MaxPooling2D(2, padding='same')
    #self.dense = Dense(128,activation='relu',kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    
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
    self.conv0 = Conv2DTranspose(16, 7, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max1 = UpSampling2D(2)
    self.conv1 = Conv2DTranspose(32, 7, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max2 = UpSampling2D(2)
    self.conv2 = Conv2DTranspose(64, 3, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    self.max3 = UpSampling2D(2)
    self.conv3 = Conv2DTranspose(1, 3, activation='relu',padding='SAME')#,kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
    


  def call(self, vals):
    x1 = self.conv0(vals[3])
    x1 = self.max1(x1) 
    x1_shape = tf.shape(vals[2])
    x1 = tf.slice(x1, tf.zeros(x1_shape.shape,dtype=tf.dtypes.int32), x1_shape)
    x1 = Concatenate()([x1, vals[2]])
    x2 = self.conv1(x1)
    x2 = self.max2(x2)
    x2_shape = tf.shape(vals[1])
    x2 = tf.slice(x2, [0,0,0,0], x2_shape)
    x2 = Concatenate()([x2, vals[1]])
    x3 = self.conv2(x2)
    x3 = self.max3(x3)
    x3_shape = tf.shape(vals[0])
    x3 = tf.slice(x3, [0,0,0,0], x3_shape)
    x3 = Concatenate()([x3, vals[0]])
    x4 = self.conv3(x3)
    return x4

class ls_layer(Model):
    def __init__(self):
        super(ls_layer, self).__init__()  
        self.fl = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.fc2 = tf.keras.layers.Dense(128, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.fc3 = tf.keras.layers.Dense(256, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.fc4 = tf.keras.layers.Dense(320, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.t_fc1 = tf.keras.layers.Dense(64, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.t_fc2 = tf.keras.layers.Dense(64, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))
        self.t_fc3 = tf.keras.layers.Dense(64, activation='relu', use_bias=True )#, kernel_regularizer=l2(reg_rate), bias_regularizer=l2(reg_rate))

    def call(self,vals):  
        x = vals[0]
        t = vals[1]
        x = self.fl(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        x = self.fc1(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        x = self.fc2(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        t = self.t_fc1(t)
        #t = tf.keras.layers.Dropout(0.5)(t)
        t = self.t_fc2(t)
        #t = tf.keras.layers.Dropout(0.5)(t)
        t = self.t_fc3(t)
        #t = tf.keras.layers.Dropout(0.5)(t)
        x2 = Concatenate()([x,t])
        x2 = self.fc3(x2)
        #x2 = tf.keras.layers.Dropout(0.5)(x2)
        x2 = self.fc4(x2)
        #x2 = tf.keras.layers.Dropout(0.5)(x2)
        return x2

class autoencoder(Model):
  def __init__(self):
    super(autoencoder, self).__init__()
    self.ae = encoder()
    self.de = decoder()
    self.ls = ls_layer()

  def call(self, vals):
    x = vals[0]
    t = vals[1]
    ae = self.ae(x)
    ls = self.ls((ae[3],t))
    ls = tf.reshape(ls, [-1,5,4,16])
    de = self.de(ae[0:3]+(ls,))
    return de

