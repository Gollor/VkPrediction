#!/usr/bin/env python

from __future__ import division

from datetime import datetime
import ConfigParser

import numpy as np
import tensorflow as tf

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

###### LOAD DATABASE ##########################################################

# Loading with a positive and negative DB
positive_data = np.load('PositiveV2data.npy') 
positive_label = np.load('PositiveV2label.npy') 
negative_data = np.load('NegativeV2data.npy') 
negative_label = np.load('NegativeV2label.npy') 
positive_size = positive_label.shape[0]
negative_data = negative_data[:positive_size]
negative_label = negative_label[:positive_size]
negative_size = negative_label.shape[0]
for i in range(positive_size):
    positive_label[i] = 1
for i in range(negative_size):
    negative_label[i] = 0
    
shuffle_in_unison(positive_data, positive_label)
shuffle_in_unison(negative_data, negative_label)
    
# Loading settings from .ini file
settings = {'Validation' : {'ValidationSize'     : 1000,
                            'NaturalDistribution': 0,
                            'TargetLabel'        : 1     },
            'Net' :        {'Tanh1Size'          : 3000,
                            'Tanh2Size'          : 500,
                            'Tanh3Size'          : 500,
                            'Tanh4Size'          : 100   },
            'Training' :   {'LearningRate'       : 0.01,
                            'LearningRateDecay'  : 0.95,
                            'EpochsAmount'       : 100,
                            'BatchSize'          : 64,
                            'DropoutRate'        : 1.0,
                            'DecayRate'          : 6e-5   }}
config = ConfigParser.ConfigParser()
config.read('KievNet.ini')
for section in config.sections():
    for option in settings[section]:
        value = settings[section][option]
        try:
            value = config.get(section, option)
        except:
            pass 
        settings[section][option] = value
print 'The next settings have being loaded: ', settings
    
    
# size from each DB
val_size = int(settings['Validation']['ValidationSize'])
half_val_size = val_size//2
percentage = positive_size / (positive_size + negative_size)
if int(settings['Validation']['NaturalDistribution']) == 0:
    percentage = 0.5 # to make equal amount of samples set percentage to 0.5 
pos_val_size = int(percentage*val_size//1)
neg_val_size = val_size - pos_val_size
val_data = np.concatenate([positive_data[-pos_val_size:],
                           negative_data[-neg_val_size:]])
val_label = np.concatenate([positive_label[-pos_val_size:],
                           negative_label[-neg_val_size:]])
positive_data = positive_data[:-pos_val_size]
positive_label = positive_label[:-pos_val_size]
negative_data = negative_data[:-neg_val_size]
negative_label = negative_label[:-neg_val_size]
positive_size -= pos_val_size
negative_size -= neg_val_size

train_size = min(positive_size, negative_size)*2

input_layer = 2**14
output_layer = 2
num_epochs = int(settings['Training']['EpochsAmount'])
batch_size = int(settings['Training']['BatchSize'])
half_batch_size = batch_size//2

data_node = tf.placeholder(
    tf.float32,
    shape=(batch_size, input_layer))
label_node = tf.placeholder(tf.int64, shape=(batch_size,))

tanh1_size = int(settings['Net']['Tanh1Size'])
tanh2_size = int(settings['Net']['Tanh2Size'])
tanh3_size = int(settings['Net']['Tanh3Size'])
tanh4_size = int(settings['Net']['Tanh4Size'])
tanh1_w = tf.get_variable("tanh1_w", shape=[input_layer, tanh1_size],
           initializer=tf.contrib.layers.xavier_initializer())
tanh1_b = tf.Variable(tf.zeros([tanh1_size]), name="tanh1_b")
tanh2_w = tf.get_variable("tanh2_w", shape=[tanh1_size, tanh2_size],
           initializer=tf.contrib.layers.xavier_initializer())
tanh2_b = tf.Variable(tf.zeros([tanh2_size]), name="tanh2_b")
tanh3_w = tf.get_variable("tanh3_w", shape=[tanh2_size, tanh3_size],
           initializer=tf.contrib.layers.xavier_initializer())
tanh3_b = tf.Variable(tf.zeros([tanh3_size]), name="tanh3_b")
tanh4_w = tf.get_variable("tanh4_w", shape=[tanh3_size, tanh4_size],
           initializer=tf.contrib.layers.xavier_initializer())
tanh4_b = tf.Variable(tf.zeros([tanh4_size]), name="tanh4_b")
soft1_w = tf.get_variable("soft1_w", shape=[tanh4_size, output_layer],
           initializer=tf.contrib.layers.xavier_initializer())
soft1_b = tf.Variable(tf.zeros([output_layer]), name="soft1_b")
           #regularizer=tf.contrib.layers.l2_regularizer(4e-4))
keep_prob = tf.placeholder(tf.float32)
    
def model (data):
    tanh1 = tf.nn.tanh(tf.matmul(data, tanh1_w) + tanh1_b)
    tanh2 = tf.nn.tanh(tf.matmul(tanh1, tanh2_w) + tanh2_b)
    tanh3 = tf.nn.tanh(tf.matmul(tanh2, tanh3_w) + tanh3_b)
    tanh4 = tf.nn.tanh(tf.matmul(tanh3, tanh4_w) + tanh4_b)
    drop1 = tf.nn.dropout(tanh4, keep_prob)
    soft1_net = tf.matmul(drop1, soft1_w) + soft1_b
    return soft1_net

logits = model(data_node)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                      logits, label_node))
regularizers = (tf.nn.l2_loss(tanh1_w) + tf.nn.l2_loss(tanh1_b) +
                tf.nn.l2_loss(tanh2_w) + tf.nn.l2_loss(tanh2_b) +
                tf.nn.l2_loss(tanh3_w) + tf.nn.l2_loss(tanh3_b) +
                tf.nn.l2_loss(tanh4_w) + tf.nn.l2_loss(tanh4_b) +
                tf.nn.l2_loss(soft1_w) + tf.nn.l2_loss(soft1_b))
loss += float(settings['Training']['DecayRate']) * regularizers
batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    float(settings['Training']['LearningRate']),      # Base learning rate.
    batch * batch_size,  # Current index into the dataset.
    train_size,          # Decay step.
    float(settings['Training']['LearningRateDecay']), # Decay rate.
    staircase=True)
optimizer = tf.train.MomentumOptimizer(learning_rate,
                                       0.9).minimize(loss,
                                                     global_step=batch)
train_prediction = tf.nn.softmax(logits)
eval_prediction = tf.nn.softmax(model(data_node))          
            
sess = tf.Session()
sess.run(tf.initialize_all_variables())

pos_p = 0
neg_p = 0
for epoch_i in range(num_epochs):
    print '*** Epoch: ' + str(epoch_i+1) + ' *** ' + datetime.strftime(datetime.now(), '%H:%M:%S')
    total_loss = 0
    batch_n = 0
    for batch_i in range(train_size // batch_size):
        pos_array = positive_data[pos_p:pos_p+half_batch_size]
        neg_array = negative_data[neg_p:neg_p+half_batch_size]
        posl_array = positive_label[pos_p:pos_p+half_batch_size]
        negl_array = negative_label[neg_p:neg_p+half_batch_size]
        batch_xs = np.zeros([batch_size, 2**14], dtype=np.float16)
        batch_ys = np.zeros((batch_size, ), dtype=np.bool)
        for i in range(64):
            if i % 2 == 0:
                batch_xs[i] = (pos_array[i//2] - 0.5)*2.0
                batch_ys[i] = posl_array[i//2]
            else:
                batch_xs[i] = (neg_array[i//2] - 0.5)*2.0
                batch_ys[i] = negl_array[i//2]
        feed_dict = {data_node: batch_xs,
                     label_node: batch_ys,
                     keep_prob: 1.0}
        _, l, lr, predictions = sess.run(
            [optimizer, loss, learning_rate, train_prediction],
            feed_dict=feed_dict)
        total_loss += l
        pos_p += half_batch_size
        if pos_p + half_batch_size > positive_size:
            pos_p = 0
            #neg_p = 0
            shuffle_in_unison(positive_data, positive_label)
            break
        neg_p += half_batch_size
        if neg_p + half_batch_size > negative_size:
            #pos_p = 0
            neg_p = 0
            shuffle_in_unison(negative_data, negative_label)
            #break
        batch_n += 1
        
    print 'Mean loss: ' + str(total_loss/batch_n)
        
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    target_label = int(settings['Validation']['TargetLabel'])#the testing label
    for batch_i in range(val_size // batch_size):
        batch_xs = np.zeros([batch_size, 2**14], dtype=np.float16)
        for i in range(batch_size):
            batch_xs[i] = (val_data[batch_i*batch_size+i] - 0.5) * 2.0
        #batch_xs = val_data[batch_i*batch_size:(batch_i+1)*batch_size][:]
        if batch_xs.shape[0] < batch_size:
            continue
        batch_ys = val_label[batch_i*batch_size:(batch_i+1)*batch_size][:]
        feed_dict = {data_node: batch_xs,
                     label_node: batch_ys,
                     keep_prob: float(settings['Training']['DropoutRate'])}
        predictions = sess.run([eval_prediction], feed_dict=feed_dict)
        for sample_i in range(batch_size):
            pred = predictions[0][sample_i].argmax() #standard
            if batch_ys[sample_i] == pred:
                if pred == target_label:
                    true_positive += 1
                else:
                    true_negative += 1
            else:
                if pred == target_label:
                    false_positive += 1
                else:
                    false_negative += 1
    total = true_positive + false_positive + true_negative + false_negative
    print 'Accuracy: ' + str((true_positive+true_negative)*100 / total) + '%'
    precision = '---'
    recall = '---'
    if false_positive+true_positive != 0:
        precision = true_positive*100 / (false_positive+true_positive)
    if false_negative+true_positive != 0:
        recall = true_positive*100 / (false_negative+true_positive)
    fmeasure = '---'
    if precision != '---' and recall != '---':
        fmeasure = 2*precision*recall / (precision+recall)
    
    print 'Precision ' + str(precision) + '%'
    print 'Recall ' + str(recall) + '%'
    print 'F-Measure ' + str(fmeasure) + '%'
