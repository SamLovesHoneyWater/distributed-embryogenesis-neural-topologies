# -*- coding: utf-8 -*-
"""

CIFAR-10 Dataset handler and related functions for CEAT

Created on Sat May 8 15:41:00 2021

@author: Sammy
"""

import pickle
import numpy as np

from HyperParams import *
X_SIZE = 2
Y_SIZE = 2
Z_SIZE = 2
OUTPUT_NUM = 2

CIFAR_PATH = 'F:\\Datasets\\cifar-10-batches-py'

# generates retina data set
def getRetinaBatch(do_reshape = False):
    data_x, data_y = [], []
    n_pix = X_SIZE * Y_SIZE
    format_specifier = '0'+str(n_pix)+'b'
    possibilities = 2**n_pix
    for left_i in range(possibilities):
        x_left = np.array(list(format(left_i, format_specifier)))
        # labels generated according to modularity paper by Clune et al
        y_left = left_i in [0, 1, 5, 4, 7, 2, 13, 8]
        for right_i in range(possibilities):
            x_right = np.array(list(format(right_i, format_specifier)))
            # labels generated according to modularity paper by Clune et al
            y_right = right_i in [0, 1, 10, 4, 14, 2, 11, 8]
            # change shape from 1D to 2D
            if do_reshape:
                x_left = x_left.reshape((X_SIZE, Y_SIZE))
                x_right = x_right.reshape((X_SIZE, Y_SIZE))
                # reshaping also affects the way left and right views are put together
                x = np.stack((x_left, x_right), axis=0)
            else:
                x = np.concatenate((x_left, x_right), axis=0)
            y = int(y_left and y_right)
            
            data_x.append(x)
            data_y.append(y)
            
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return(data_x, data_y, None)

'''
# reads cifar-10 file as dict
def unpickle(file):
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict
    
# gets a cifar-10 data batch (one among the five available)
def getCifarBatch(batch_num, do_reshape = False, do_std = True):
    raw = unpickle(CIFAR_PATH + '\\data_batch_' + str(batch_num))
    
    # process raw data
    data_x = np.array(raw[b'data'])
    data_y = np.array(raw[b'labels'])
    data_batch_n = np.array(raw[b'batch_label'].decode())
    
    # check data shape
    if data_x.shape != (10000, 3072):
        print('\n[WARNING] Raw data shape not as expected!\n')
    
    # reshape image data if necessary
    if do_reshape:
        data_x = data_x.reshape([10000, 3, 32, 32])
    
    if do_std:
        data_x = data_x/255
    
    return(data_x, data_y, data_batch_n)

#_, y, __ = getCifarBatch(1)
#print(y[0])
'''

def getInputNodePos(i):
    z = i // (X_SIZE * Y_SIZE)
    i = i % (X_SIZE * Y_SIZE)
    y = i // X_SIZE + (SPACE_Y - Y_SIZE)//2
    x = i % X_SIZE + (SPACE_X - X_SIZE)//2
    return (x,y,z)

def getOutputNodePos(i):
    x = i + (SPACE_X - OUTPUT_NUM)//2
    y = SPACE_Y // 2
    z = SPACE_Z - 1
    return (x-2,y-2,z)

def choice2onehot(choice):
    # single input
    if type(choice) == np.int32:
        y = [0 for i in range(OUTPUT_NUM)]
        y[choice] = 1
    # multiple data inputs
    else:
        y = [[0 for i in range(OUTPUT_NUM)] for j in range(len(choice))]
        for i in range(len(choice)):
            y[i][choice[i]] = 1
    return y
