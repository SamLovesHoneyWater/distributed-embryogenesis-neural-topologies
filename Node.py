# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:11:06 2021

@author: Sammy
"""

import tensorflow as tf 
from HyperParams import *
######################
import time
######################

class Node(object):
    def __init__(self, connections, coordinates, bias = 0, \
                 activation = 'relu',\
                 node_type = 'hidden',
                 io_index = None):
        self.sources = [c[0] for c in connections[0]]
        self.w = [c[1] for c in connections[0]]
        #self.incoming = connections[0]      # [[source_node, weight], ...]
        self.outgoing = connections[1]      # [target_node, ...]
        self.coords = coordinates
        self.b = bias
        self.activation = getActivation(activation)
        self.node_type = node_type
        self.markings = ''
        self.io_index = io_index
    
    def hatch(self):
        #t0 = time.time()
        #self.sources = [c[0] for c in self.incoming]
        #self.w = tf.Variable([c[1] for c in self.incoming], dtype = STD_DTYPE)
        self.w = tf.Variable(self.w, dtype = STD_DTYPE)
        self.b = tf.Variable(self.b, dtype = STD_DTYPE)
        self.fed = False
        
        #del self.outgoing
        
        #print(time.time()-t0)

    # feed forward for a single node
    def ff1(self):
        if self.fed:
            return
        if self.sources == []:
            return
        x_list = []
        for node in self.sources:
            if not node.fed:
                a = node.ff1()
                x_list.append(a)
            else:
                x_list.append(node.a)
        x = x_list
        
        #weighted = tf.multiply(x, self.w)
        #weighted_sum = tf.reduce_sum(weighted)
        weighted_sum = tf.tensordot(self.w, x, axes = 1)
        
        
        z = tf.add(weighted_sum, self.b)
        a = self.activation(z)
        self.a = a
        self.fed = True
        return a
            
    # used to keep the topology a DAG
    def isFatherOf(self, node0):
        if node0 in self.outgoing:
            return True
        if len(self.outgoing) == 0:
            return False
        for node in self.outgoing:
            if(node.isFatherOf(node0)):
                return True
        return False
                
    
'''
############################
# not used
def getX(node):
    if not node.fed:
        node.ff1()
    return node.a
############################
'''

def getActivation(a):
    if a == 'relu':
        return tf.nn.relu
    if a == 'sigmoid':
        return tf.nn.sigmoid
    if a == 'tanh':
        return tf.nn.tanh
    if a == 'softmax':
        return tf.nn.softmax
    if a == 'linear':
        return linear

def linear(z):
    return z