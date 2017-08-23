
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train



# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.


#%%

import tensorflow as tf

#%%
def inference(images, batch_size, n_classes):

    conv1=tf.layers.conv2d(
            inputs=images,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        
    conv2=tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        
    conv3=tf.layers.conv2d(
          inputs=pool2,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        
    flatten=tf.reshape(pool3, [-1, 26 * 26 * 64])
    dense1 = tf.layers.dense(inputs=flatten, 
                          units=1024, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2= tf.layers.dense(inputs=dense1, 
                          units=512, 
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits= tf.layers.dense(inputs=dense2, 
                            units=n_classes, 
                            activation=None,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        
    return logits

#%%

def losses(logits, labels):
    
    loss1=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    loss = tf.reduce_mean(loss1)
    return loss

#%%
def trainning(loss, learning_rate):
    
    train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    return train_op

#%%
def evaluation(logits, labels):

    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), labels)    
    accuracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy

#%%




