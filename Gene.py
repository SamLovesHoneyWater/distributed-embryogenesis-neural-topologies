# -*- coding: utf-8 -*-
"""
Created on Thu May 6 17:11:23 2021

@author: Sammy
"""

import random
from HyperParams import *

OP_DICT = {'AddConn':0, 'InsertNode':1}
OP_LIST = ['AddConn', 'InsertNode']

HM_DICT = {'Void':0, 'AddConn':1,'InsertNodeParent':2, 'InsertNodeChild':3}
HM_LIST = {'Void', 'AddConn','InsertNodeParent', 'InsertNodeChild'}

class Gene(object):
    def __init__(self, operator, own_historical_marking,\
                 target_historical_marking = None,\
                 uses_before_expiration = 1,\
                 new_connection_weight = None):
        
        # gene parameters
        self.op = operator
        self.own_hm = own_historical_marking
        self.exp_count = uses_before_expiration
        self.ever_activated = False
        
        # add connection operator
        if operator == 0:
            self.target_hm = target_historical_marking
            self.w = new_connection_weight
            if (self.target_hm is None) or (self.w is None):
                self.raiseParamError()
            
        # insert node operator
        if operator == 1:
            # swaps source/target to facilitate search
            self.target_hm = own_historical_marking
            self.own_hm = target_historical_marking
            self.w = new_connection_weight
            if (self.own_hm is None) or (self.w is None):
                self.raiseParamError()
    
    def getClone(self):
        if self.op == 0:
            return Gene(self.op, self.own_hm, \
                        target_historical_marking = self.target_hm, \
                            uses_before_expiration = self.exp_count, \
                                new_connection_weight = self.w)
        if self.op == 1:
            return Gene(self.op, self.target_hm, \
                        target_historical_marking = self.own_hm, \
                            uses_before_expiration = self.exp_count, \
                                new_connection_weight = self.w)
    
    # called by an op0 gene to create a subsequent gene that inserts
    # a node on the connection formed by the op0 gene
    def getInsGene(self):
        if self.op != 0:
            self.raiseInsGeneError()
        new_gene =  Gene(1, self.own_hm + '1', \
                         target_historical_marking = self.target_hm + '1', \
                             uses_before_expiration = self.exp_count, \
                                 new_connection_weight = 1)
        weight_type = type(INSERT_NODE_WEIGHT)
        # constant value
        if weight_type == type(1) or weight_type == type(1.1):
            new_gene.w = INSERT_NODE_WEIGHT
        # else, random value
        else:
            new_gene = new_gene.mutateWeight()
        return new_gene
    
    def mutateActivity(self):
        new_gene = self.getClone()
        if random.random() < PROB_SMOOTH_ACTIVITY_CHANGE:
            # normal addition to expiration counts
            d_a = int(random.normalvariate(0, MORE_ACTIVITY_SIGMA))
            new_gene.exp_count = max(1, new_gene.exp_count + d_a)  # no smaller than 1
            new_gene.exp_count = min(new_gene.exp_count, EXP_COUNT_CAP)  # no greater than cap
        else:
            # uniform randomness
            new_gene.exp_count = random.randint(1, EXP_COUNT_CAP)
        new_gene.activity = new_gene.exp_count
        return new_gene
        
    def mutateWeight(self):
        new_gene = self.getClone()
        new_gene.w = random.normalvariate(0, NEW_WEIGHT_SIGMA)
        return new_gene
    
    def mutateOwnHM(self, new_hm):
        if self.op == 0:
            self.own_hm = new_hm
        elif self.op == 1:
            self.target_hm = new_hm

    def mutateTargetHM(self, new_hm):
        if self.op == 0:
            self.target_hm = new_hm
        elif self.op == 1:
            self.own_hm = new_hm

    def printInfo(self):
        prnt_str = '\t>>\tOperator: '
        prnt_str += str(self.op)
        prnt_str += '\tOwn Marking: '
        prnt_str += str(self.own_hm)
        if self.op in [0, 1]:
            prnt_str += '\tTarget Marking: '
            prnt_str += str(self.target_hm)
        prnt_str += '\tWeight: '
        prnt_str += str(self.w)
        prnt_str += '\tActivity: '
        prnt_str += str(self.exp_count)
            
        print(prnt_str)
    
    # raised when wrong params are fed when initializing a gene object
    def raiseParamError(self):
        print('\t>>\tWARNING: Operator-specific parameters not fully\
              specified when initializing a gene with operator', self.op)
    
    # raised when attempting to create an op1 gene that is not based on an op0 gene 
    def raiseInsGeneError(self):
        print('\t>>\tWARNING: Failed to create inserting gene with\
              respect to original gene with operator', self.op)
              
              
        ''' # not used
    def isApplicable(self, node):
        # incorrect historical markings on node
        if node.markings != self.own_hm:
            return False
        '''
        '''
        # not used
        # incorrect number of incoming/outgoing connections
        if [ len(node.incoming), len(node.outgoing) ] != io_pair:
            return False
        '''
        '''
        # all conditions met
        return True
'''
        
        