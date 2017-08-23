#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:12:04 2017

@author: user
"""

from skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time


#path_validation='/home/user/Desktop/flower-tensorflow/validation'

#将所有的图片resize成100*100
w=100
h=100
c=3
image_W=100
image_H=100
batch_size=32
capacity=64
#读取图片
def get_files(path):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    daisy=[]
    dandelion=[]
    roses=[]
    sunflowers=[]
    tulips=[]
    label_daisy=[]
    label_dandelion=[]
    label_roses=[]
    label_sunflowers=[]
    label_tulips=[]
    for file in os.listdir(path):
        print file
        if file=='daisy':
            for files in os.listdir(path+file+'/'):
                daisy.append(path+file+'/'+files)
                label_daisy.append(0)           
        elif file=='roses':
            for files in os.listdir(path+file+'/'):
                roses.append(path+file+'/'+files)
                label_roses.append(1) 
               
        elif file=='sunflowers':
            for files in os.listdir(path+file+'/'):
                sunflowers.append(path+file+'/'+files)
                label_sunflowers.append(2) 
        elif file=='dandelion':
            for files in os.listdir(path+file+'/'):
                dandelion.append(path+file+'/'+files)
                label_dandelion.append(3) 
               
        else:
             for files in os.listdir(path+file+'/'):
                 tulips.append(path+file+'/'+files)
                 label_tulips.append(4) 
               
               
 
    
    image_list = np.hstack((daisy,roses,sunflowers,dandelion,tulips))
    label_list = np.hstack((label_daisy,label_roses,label_sunflowers,label_dandelion,label_tulips))
    
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    return image_list, label_list

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''
    
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

  
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
   
    
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label],
                                                batch_size= batch_size,
                                                num_threads= 64, 
                                                capacity = capacity)
    
    #you can also use shuffle_batch 
#    image_batch, label_batch = tf.train.shuffle_batch([image,label],
#                                                      batch_size=BATCH_SIZE,
#                                                      num_threads=64,
#                                                      capacity=CAPACITY,
#                                                      min_after_dequeue=CAPACITY-1)
    
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch
