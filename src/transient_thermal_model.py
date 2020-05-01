#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose, Concatenate
from tensorflow.keras import Model, regularizers
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from time import time
import matplotlib.animation as animation
import os
import yaml
from encoder_decoder import autoencoder
import shutil
import argparse

# x_data is input image frame, t_data is time, y_data is golden output image frame
def train_model(x_data, t_data, y_data):
    # Create an instance of the model
    model = autoencoder()
    initial_learning_rate = 0.0005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.98,
        staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss='mse',
                  metrics=['mse', 'mae', 'mape'])
    
    st = time()
    history = model.fit(x=[x_data,t_data],y=y_data, epochs=100,
                validation_split=0.1,
                shuffle=True)
    et = time()
    tt = et-st
    print("Elapsed time: %03d:%02d:%05.2f"%(int(tt/3600),int(tt/60)%60,tt%60))
    return model



def predict_temperature(model, normalization_data, test_data_dir):
    data_runs = [f for f in glob(test_data_dir+"**/*", recursive=False)]
    
    power_map = np.zeros((len(data_runs),45,34,32))
    temp_map = np.zeros((len(data_runs),45,34,32))
    
    num_test_data = 0;
    for i, run in enumerate(data_runs):
        print("Reading data: ",run)
        run_num = re.findall(r'\d+', run)
        for frame in range(45):
            fname = run+"/ml_raw_data_%d_%d.csv"%(int(run_num[0]),frame+1)
            power_map[i,frame,...],temp_map[i,frame,...] = read_image(fname)
        num_test_data+=1

    max_x = normalization_data[0] 
    min_x = normalization_data[1]
    max_t = normalization_data[2]
    min_t = normalization_data[3]
    max_y = normalization_data[4]
    min_y = normalization_data[5]
    Writer = animation.writers['html']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    im = []
    figs= []
    anis = []
    num_im = num_test_data
    predicted_video = np.zeros((num_im,45,34,32))
    err_video = np.zeros((num_im,45,34,32))
    #get_ipython().run_line_magic('matplotlib', 'notebook')
    for im_num in range(num_im):  
        def updatefig1(frame,im_num):
            im[(im_num)*3].set_array(temp_map[im_num,frame,...])
            im[(im_num)*3+1].set_array(predicted_video[im_num,frame,...])
            
        def updatefig2(frame,im_num):
            im[(im_num)*3+2].set_array(err_video[im_num,frame,...])
    
        for frame in range(45):
            data_pt = np.array([[frame]])/max_t
            in_data = power_map[im_num:im_num+1,0,...,np.newaxis]
            in_data = (in_data-min_x)/(max_x-min_x)
            predicted = model.predict((in_data,data_pt))
            predicted_video[im_num,frame,...] = (np.squeeze(predicted)*(max_y-min_y))+min_y
            err_video[im_num,frame,...] = predicted_video[im_num,frame,...] - temp_map[im_num,frame,...] 
        print("Writing result of test: ", data_runs[im_num])
        test_data_num = re.findall(r'\d+', data_runs[im_num])
        test_data_num = int(test_data_num[0])
        fig, axes = plt.subplots(1,2)
        max_val = np.max((np.max(temp_map[im_num,...]), np.max(predicted_video[im_num,...])))
        max_err = np.max(err_video[im_num,...])
        min_err = np.min(err_video[im_num,...])
        im.append(axes[0].imshow(temp_map[im_num,0], vmin=0, vmax=max_val))
        im.append(axes[1].imshow(predicted_video[im_num,0,...], vmin=0, vmax=max_val))
        fig.colorbar(im[3*(im_num)+1], ax=axes.ravel().tolist())
        
        anis.append(animation.FuncAnimation(fig, updatefig1, frames=range(45), fargs=(im_num,),
                                      interval=150 ))
        anis[2*(im_num)].save("output_plots/run_contours_%d.html"%test_data_num, writer=writer)
        plt.close()
        fig2 = plt.figure()
        im.append(plt.imshow(err_video[im_num,0,...], vmin=min_err, vmax=max_err))
        plt.colorbar()
        anis.append(animation.FuncAnimation(fig2, updatefig2, frames=range(45), fargs=(im_num,),
                                      interval=150 ))
        anis[2*(im_num)+1].save("output_plots/err_contours_%d.html"%test_data_num, writer=writer)
        plt.close()
        dir = os.path.join("./","output_plots","output_plots")
        if not os.path.exists(dir):
            os.mkdir(dir)
            os.rename("output_plots/run_contours_%d_frames"%test_data_num,"output_plots/output_plots/run_contours_%d_frames"%test_data_num )
            os.rename("output_plots/err_contours_%d_frames"%test_data_num,"output_plots/output_plots/err_contours_%d_frames"%test_data_num )
        else:
            if (os.path.exists("output_plots/output_plots/run_contours_%d_frames"%test_data_num)
            and os.path.exists("output_plots/output_plots/err_contours_%d_frames"%test_data_num)):
                shutil.rmtree("output_plots/output_plots/run_contours_%d_frames"%test_data_num )
                shutil.rmtree("output_plots/output_plots/err_contours_%d_frames"%test_data_num )
            os.rename("output_plots/run_contours_%d_frames"%test_data_num,"output_plots/output_plots/run_contours_%d_frames"%test_data_num )
            os.rename("output_plots/err_contours_%d_frames"%test_data_num,"output_plots/output_plots/err_contours_%d_frames"%test_data_num )



def read_data(data_dir):
    data_runs = [f for f in glob(data_dir+"**/*", recursive=False)]
    
    power_map = np.zeros((len(data_runs),45,34,32))
    temp_map = np.zeros((len(data_runs),45,34,32))
    
    num_images = 0;
    for i, run in enumerate(data_runs):
        print("Reading data: ",run)
        run_num = re.findall(r'\d+', run)
        run_num = int(run_num[0])
        for frame in range(45):
            print(run)
            fname = run+"/ml_raw_data_%d_%d.csv"%run_num,frame+1)
            print(fname)
            power_map[i,frame,...],temp_map[i,frame,...] = read_image(fname)
            num_images+=1
    
    
    count = 0
    x_data = np.zeros((num_images,34,32,1))
    y_data = np.zeros((num_images,34,32,1))
    t_data = np.zeros((num_images,1))
    for im_num in range(len(data_runs)):
        for frame in range(45):
            x_data[count,:,:,0] = power_map[im_num,44,...]
            y_data[count,:,:,0] = temp_map[im_num,frame,...]
            t_data[count,...] = frame
            count +=1
    return [x_data, y_data, t_data]
    
def process_data(x_data, y_data, t_data):
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    
    x_data = x_data[indices]
    y_data = y_data[indices]
    t_data = t_data[indices]
    
    min_x = np.min(x_data)
    max_x = np.max(x_data)
    x_data = (x_data-min_x)/(max_x-min_x)
    
    max_t =44
    min_t = 0
    t_data = (t_data)/max_t
    
    min_y=np.min(y_data)
    max_y = np.max(y_data)
    y_data = (y_data-min_y)/(max_y-min_y)
    normalization_data = [max_x, min_x, max_t, min_t, max_y, min_y]
    return x_data, y_data, t_data, normalization_data


def read_image(fname):
   power_map = np.zeros((34,32))
   temp_map = np.zeros((34,32))
   with open(fname) as csvfile:
       readCSV = csv.reader(csvfile, delimiter=',')
       for row in readCSV:
           x = int(np.round(float(row[1])/2.5e-4))
           y = int(np.round(float(row[2])/2.5e-4))
           dyn_pow = float(row[3])
           leak_pow = float(row[4])
           alpha = float(row[5])
           power_map[x,y] = alpha*dyn_pow + leak_pow
           temp_map[x,y] = float(row[7])
   return power_map,temp_map

stream = open("config.yaml", 'r')
config = yaml.load(stream)
print ("Using the following settings:")
for key, value in config.items():
    print (key + " : " + str(value))




def main():
    
    parser = argparse.ArgumentParser(description="Training and inference flow for transient thermal analysis")
    parser.add_argument("-train_data_path",
                        help="Provide path to the directory containing Transient_runs and Steady_run folders with training data",
                        required=True)
    parser.add_argument("-test_data_path",
                       help="Provide path to the directory containing Transient_runs and Steady_run folders with test data",
                       required=True)
    parser.add_argument("-output_plot_dir",
                        help="Provide path to generate the output plots",
                       required=True)
    args = parser.parse_args()
    train_data_dir =  args.train_data_path
    test_data_dir =  args.test_data_path
    output_plot_dir = args.output_plot_dir
    
    data_train = read_data(train_data_dir)
    #data_test = read_data(test_data_dir)
    x_data, y_data, t_data, normalization_data = process_data(data_train[0],
            data_train[1], data_train[2])
    model = train_model(x_data, t_data, y_data)
    predict_temperature(model, normalization_data, test_data_dir)
    #plot_test_results(predicted_video, err_video, temp_map, output_plot_dir)
    


if __name__ == '__main__':
    main()

   

