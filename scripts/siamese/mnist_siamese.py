#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy as sp
import random
import os
import itertools
import pickle
import matplotlib.pyplot as plt

from arch.siamese.siamese_graph import mnist_siamese
from util.eval import load_mnist_data
from util.normalization import global_mean_std
from util.misc import tuple_list_find
from arch.misc import ExponentialDecay
from arch.io import save_variables
from util.batch import random_batch_generator, batch_generator
from util.visualization import load_palette
from util.transform import RandomizedTransformer, Affine
from config import temp_folder


#https://github.com/ppwwyyxx/tensorpack/blob/master/examples/SimilarityLearning/mnist-embeddings.py
#https://github.com/ardiya/siamesenetwork-tensorflow/blob/master/model.py

def contrastive_loss(x1, x2, y, margin=0.2, eps=1e-12):
    #http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    y = tf.cast(y, tf.float32)
    Dw = tf.sqrt(tf.reduce_sum(tf.pow(x1-x2, 2), axis=1, keepdims=True)+eps) # euclidean distance, eq 1
    Ls = tf.square(Dw)*0.5 # loss for similar images, eq 4
    Ld = tf.square(tf.maximum(0.0, margin-Dw))*0.5 # loss for dissimilar images, eq 4
    pair_loss = (1.0-y)*Ls + y*Ld # eq 3
    loss = tf.reduce_mean(pair_loss) # eq 2
    return loss


def generate_pair_indices(x, y, max_pos = 100, max_neg = 100, seed = 42):
    n_classes = y.shape[1]    

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    label_index = dict()
    for idx, _label in enumerate(y):
        label = np.argmax(_label)
        if label in label_index:
            label_index[label].append(idx)
        else:
            label_index[label] = [idx]
    
    x1 = []
    x2 = []
    yp = []
    
    # select positive (similar) pairs
    max_pos_perc_class = np.round(max_pos / (1.0*n_classes)).astype(int)
    for i in range(n_classes):
        n_comb = np.int64(sp.special.binom(len(label_index[i]), 2))
        n = min(n_comb, max_pos_perc_class)
        sels = list(itertools.combinations(label_index[i], 2))
        sel_idx = np.random.choice(range(n_comb), n)
        for s in sel_idx:
            x1.append(sels[s][0])
            x2.append(sels[s][1])
        yp = yp + [[0]]*n
            
    # select negative (dissimilar) pairs
    n_class_comb = sp.special.binom(n_classes, 2) # 45 for 10 classes
    max_neg_per_class_comb = np.round(max_neg / (1.0*n_class_comb)).astype(int)
    for i,j in itertools.combinations(range(n_classes), 2):            
        n1 = min(len(label_index[i]), max_neg_per_class_comb)
        n2 = min(len(label_index[j]), max_neg_per_class_comb)
        n = min(n1, n2)
        sel1 = np.random.choice(label_index[i], n)
        sel2 = np.random.choice(label_index[j], n)
        for k in range(n):
            x1.append(sel1[k])
            x2.append(sel2[k])
            yp.append([1])
    
    return np.array([x1, x2]).T, np.array(yp).astype(np.float32)


def generate_pair_indices2(x, y, max_pos = 100, max_neg = 100, seed = 42):
    n_classes = y.shape[1]    

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    label_index = dict()
    for idx, _label in enumerate(y):
        label = np.argmax(_label)
        if label in label_index:
            label_index[label].append(idx)
        else:
            label_index[label] = [idx]
    
    x1 = []
    x2 = []
    yp = []
    class_sels = dict()
    
    # select positive pairs
    max_pos_perc_class = np.round(max_pos / (1.0*n_classes)).astype(np.int64)
    for i in range(n_classes):
        n_comb = np.int64(sp.special.binom(len(label_index[i]), 2))
        n = min(n_comb, max_pos_perc_class)
        sels = list(itertools.combinations(label_index[i], 2))
        sel_idx = np.random.choice(range(n_comb), n)
        x1_sels = []
        x2_sels = []
        for s in sel_idx:
            x1_sels.append(sels[s][0])
            x2_sels.append(sels[s][1])
        x1 = x1 + x1_sels
        x2 = x2 + x2_sels
        yp = yp + [[0]]*n
        curr_sels = x1_sels + x2_sels
        curr_sels = list(set(curr_sels))
        class_sels[i] = curr_sels
            
    # select negative pairs
    n_class_comb = sp.special.binom(n_classes, 2) # 45 for 10 classes
    max_neg_per_class_comb = np.round(max_neg / (1.0*n_class_comb)).astype(np.int64)
    for i,j in itertools.combinations(range(n_classes), 2):
        label_index_i = class_sels[i]
        label_index_j = class_sels[j]
        n1 = min(len(label_index_i), max_neg_per_class_comb)
        n2 = min(len(label_index_j), max_neg_per_class_comb)
        n = min(n1, n2)
        sel1 = np.random.choice(label_index_i, n).tolist()
        sel2 = np.random.choice(label_index_j, n).tolist()
        x1 = x1 + sel1
        x2 = x2 + sel2
        yp = yp + [[1]]*n
    
    return np.array([x1, x2]).T, np.array(yp).astype(np.float32)



def generate_pair_indices3(x, y, max_pos = 100, max_neg = 100, seed = 42):
    n_classes = y.shape[1]    

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    label_index = dict()
    for idx, _label in enumerate(y):
        label = np.argmax(_label)
        if label in label_index:
            label_index[label].append(idx)
        else:
            label_index[label] = [idx]
    
    x1 = []
    x2 = []
    yp = []
    class_sels = dict()
    
    # select positive pairs
    max_pos_perc_class = np.round(max_pos / (1.0*n_classes)).astype(np.int64)
    for i in range(n_classes):
        n_comb = np.int64(sp.special.binom(len(label_index[i]), 2))
        n = min(n_comb, max_pos_perc_class)
        sels = list(itertools.combinations(label_index[i], 2))
        sel_idx = np.random.choice(range(n_comb), n)
        x1_sels = []
        x2_sels = []
        for s in sel_idx:
            x1_sels.append(sels[s][0])
            x2_sels.append(sels[s][1])
        x1 = x1 + x1_sels
        x2 = x2 + x2_sels
        yp = yp + [[0]]*n
        curr_sels = x1_sels + x2_sels
        curr_sels = list(set(curr_sels))
        class_sels[i] = curr_sels
            
    # select negative pairs
    all_sels = []
    for i in range(0,n_classes-1):
        for j in range(i+1,n_classes):
            for ci in class_sels[i]:
                for cj in class_sels[j]:
                    all_sels.append((ci, cj))
    n = min(len(all_sels), max_neg)
    sel_idx = np.random.choice(range(len(all_sels)), n).tolist()
    all_sels = [all_sels[s] for s in sel_idx]
    for sel in all_sels:
        x1.append(sel[0])
        x2.append(sel[1])
    yp = yp + [[1]]*n
    
    
    return np.array([x1, x2]).T, np.array(yp).astype(np.float32)



def main(seed = 42):
    weight_decay = 0.0001
    margin = 0.2
    n_epochs = 50
    batch_size = 128
    
    _tr_x, _tr_y, _te_x, _te_y = load_mnist_data()
    _tr_x, _te_x = global_mean_std(_tr_x, _te_x)
    
    
    
    temp_path = os.path.join(temp_folder, "mnist_siamese_data.pkl")
    if not os.path.exists(temp_path):
#        tr_x, tr_y = generate_pair_indices(_tr_x, _tr_y, max_pos = 9000, max_neg = 9000, seed = seed)
        tr_x, tr_y = generate_pair_indices2(_tr_x, _tr_y, max_pos = 45000, max_neg = 45000, seed = seed)
#        te_x, te_y = generate_pair_indices(_te_x, _te_y, max_pos = 2250, max_neg = 2250, seed = seed)
        te_x, te_y = generate_pair_indices2(_te_x, _te_y, max_pos = 9000, max_neg = 9000, seed = seed)
        with open(temp_path, "wb") as f:
            pickle.dump((tr_x, tr_y, te_x, te_y), f)
    else:
        with open(temp_path, "rb") as f:
            tr_x, tr_y, te_x, te_y = pickle.load(f)
    
    
    height = _tr_x.shape[1]
    width = _tr_x.shape[2]
    n_chans = _tr_x.shape[3]
    n_classes = _tr_y.shape[1]    
    

    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)

    x1 = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input1")
    x2 = tf.placeholder(tf.float32, [None, height, width, n_chans], name="input2")
    label = tf.placeholder(tf.float32, [None, 1], name='label')
    
    layers, variables = mnist_siamese(x1, x2, weight_decay = weight_decay, seed = seed)
    training = tuple_list_find(variables, "training")[1]
    output = tuple_list_find(layers, "output")[1]
    output1 = tuple_list_find(output[0], "output")[1]
    output2 = tuple_list_find(output[1], "output")[1]
    
    _loss = contrastive_loss(output1, output2, label, margin=margin)
    if weight_decay is not None:
        reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_fn = _loss + tf.reduce_sum(reg_ws)    
    else:
        loss_fn = _loss
    
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.99,
                use_nesterov=True).minimize(loss_fn, global_step = global_step)
        
    # learning rate with exponential decay
    lr_decay = ExponentialDecay(start=0.01, stop=0.001, max_steps=50)
        

    # data augmentation     
#    transformer = RandomizedTransformer(
#            transformer_class = Affine,
#            params = [('shape', (height, width, n_chans)),
#                      ('scale', 1.0)],
#            rand_params = [('r', [-3.0, 3.0]),
#                           ('tx', [-3.0, 3.0]),
#                           ('ty', [-3.0, 3.0]),
#                           ('reflect_y', [False, True])],
#            mode = 'each',
#            random_seed = seed)
    transformer = None
        
    session = tf.Session()
    with session.as_default():
        session.run(tf.global_variables_initializer())
        for i in range(n_epochs):
            lr = next(lr_decay)
            # training via random batches
            for (xb, yb) in random_batch_generator(batch_size, tr_x, tr_y, seed = seed+i):
                idx1 = xb[:,0]
                idx2 = xb[:,1]
                xb1 = _tr_x[idx1]
                xb2 = _tr_x[idx2]
                if transformer is not None:
                    xbtr1 = np.zeros_like(xb1)
                    for j in range(len(xb1)):
                        xbtr1[j] = transformer.transform(xb1[j])
                    xbtr2 = np.zeros_like(xb2)
                    for j in range(len(xb2)):
                        xbtr2[j] = transformer.transform(xb2[j])
                else:
                    xbtr1 = xb1
                    xbtr2 = xb2
                session.run(train_step, feed_dict = {x1: xbtr1,
                                                     x2: xbtr2,
                                                     label: yb,
                                                     training: True,
                                                     learning_rate: lr})

            tr_loss = []
            # evaluations on train set
            for (xb, yb) in batch_generator(256, tr_x, tr_y, fixed_size = False):
                idx1 = xb[:,0]
                idx2 = xb[:,1]
                xb1 = _tr_x[idx1]
                xb2 = _tr_x[idx2]
                loss = session.run(_loss, feed_dict = {x1: xb1,
                                                       x2: xb2,
                                                       label: yb,
                                                       training: False})
                tr_loss.append(loss)    
            
            
            te_loss = []
            # evaluations on test set
            for (xb, yb) in batch_generator(256, te_x, te_y, fixed_size = False):
                idx1 = xb[:,0]
                idx2 = xb[:,1]
                xb1 = _te_x[idx1]
                xb2 = _te_x[idx2]
                loss = session.run(_loss, feed_dict = {x1: xb1,
                                                       x2: xb2,
                                                       label: yb,
                                                       training: False})
                te_loss.append(loss)    


            print("Epoch: ", i)
            print("Learning rate: ", lr)
            print("Train loss: " + str(np.mean(tr_loss)))
            print("Test loss: " + str(np.mean(te_loss)))            


        var_path = os.path.join(temp_folder, "mnist_siamese_net.h5")
        save_variables(session, var_path)

        embeddings = []
        labels = []
        for xb in batch_generator(256, tr_x, y=None, fixed_size = False):
            idx1 = xb[:,0]
            idx2 = xb[:,1]
            xb1 = _tr_x[idx1]
            xb2 = _tr_x[idx2]
            yb1 = _tr_y[idx1]
            yb2 = _tr_y[idx2]
            out1 = session.run(output1, feed_dict = {x1: xb1, training: False})
            out2 = session.run(output1, feed_dict = {x1: xb2, training: False})
            embeddings.append(out1)
            embeddings.append(out2)
            labels.append(yb1)
            labels.append(yb2)
    
    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    labels = np.argmax(labels, axis=1)

    # generated by glasbey
    palette = load_palette("palette50.txt", skip_first=True, scale01=True) # skip white
    palette = palette[0:n_classes]
    #palette = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    f = plt.figure()
    for i in range(n_classes):
        idx = labels == i
        coords = embeddings[idx,:]
        plt.plot(coords[:,0], coords[:,1], '.', color=palette[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.show()

if __name__ == "__main__":
    main()
