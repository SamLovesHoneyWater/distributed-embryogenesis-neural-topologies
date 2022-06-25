# -*- coding: utf-8 -*-
"""
Created on Sat May 8 23:56:00 2021

@author: Sammy
"""

# topology grid params (not data)
SPACE_X = 10
SPACE_Y = 10
SPACE_Z = 10

MAX_CONN_LEN = 99999999 # maximum length during connection search (not actually used)

# final layer activation
OUTPUT_LAYER_ACTIVATION = 'linear' # 'linear' if softmax-masked, 'sigmoid' if sigmoid output
FINAL_SOFTMAX_MASK = True # true if output layer is softmax

# learning rate (exponential decay, r_n = r_0 * kappa^n)
LEARNING_RATE = .1
LEARNING_RATE_DECAY = .95
STD_BATCH_SIZE = 5000

STD_DTYPE = 'float32' # tf dtype for network variables
EXP_COUNT_CAP = 16    # maximum number of allowed gene reuses

MAX_EMBRYO_TIME = 5   # embrogeny timeout (in seconds)

##################################################################################
########                                                                  ########
########                       Evolution params                           ########
########                                                                  ########
##################################################################################

HAPPY_NUMBER = 24

INITIAL_POP_SIZE = 100
STD_POP_SIZE = INITIAL_POP_SIZE
INSERT_NODE_WEIGHT = 'random'     # weight for posterior connection after node insertion
                            # anterior connection has original connection weight
                            # only used for a insert-node mutations
PROB_REPLICATE_SELF = 0.15 # independent
PROB_NEW_RANDOM_CONN = 0.3 # independent

#############################
###### inter-dependent ######
# MAKE SURE SUM IS NOT GREATER THAN ONE!!!
PROB_CHANGE_SELF = 0.3
PROB_DEL_SELF = 0.2
# non-mutative inheritance probability = 1 - PROB_CHANGE_SELF - PROB_DEL_SELF
###### inter-dependent ######
#############################

#############################
###### inter-dependent ######
# MAKE SURE SUM IS EXACTLY ONE!!!
PROB_CHANGE_WEIGHT = .15
PROB_INS_NODE = .4
PROB_CHANGE_ACTIVITY = .15
PROB_CHANGE_OWN_HM = .15
PROB_CHANGE_TARGET_HM = .15
###### inter-dependent ######
#############################

PROB_SMOOTH_ACTIVITY_CHANGE = .8   # change exp_count w/r to current value, subject to bounds
                                    # otherwise, uniform random value btwn bounds
MORE_ACTIVITY_SIGMA = 4 # used to augment exp_count smoothly, miu = 0
NEW_WEIGHT_SIGMA = 20  # used to randomize weight, miu = 0
