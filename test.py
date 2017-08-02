#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 16:30:21 2017

@author: user
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%% Evaluate one image
# when training, comment the following codes.


#from PIL import Image
#import matplotlib.pyplot as plt
#
def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]

    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image

def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    
    # you need to change the directories to yours.
    train_dir = '/home/user/Desktop/flower-tensorflow/train/'
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 5
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/user/Desktop/flower-tensorflow/train_logits/'
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            print prediction
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a daisy with possibility %.6f' %prediction[:, 0])
            elif max_index==1:
                print('This is a roses with possibility %.6f' %prediction[:, 1])
            elif max_index==2:
                print('This is a sunflowers with possibility %.6f' %prediction[:, 2])
            elif max_index==3:
                print('This is a dandelion with possibility %.6f' %prediction[:, 3])
            else:
                print('This is a tuplits with possibility %.6f' %prediction[:, 4])

#%%
