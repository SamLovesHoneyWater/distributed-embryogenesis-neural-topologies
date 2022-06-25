# -*- coding: utf-8 -*-
"""
Created on Sun May 23 22:55:31 2021

@author: Sammy
"""

from Agent import *

# spawn a list of minimal networks
def spawnMinimal(amount):
    agents = []
    for i in range(amount):
        genes = []
        for j in range(1, 3072):
            g = Gene(0, str(-j), target_historical_marking = str(-32*32*3-(j-1)%10-1), \
                     new_connection_weight = 1)
            genes.append(g)
            g = Gene(1, str(-j), target_historical_marking = str(-32*32*3-(j-1)%10-1))
            genes.append(g)
        '''
        # genes of all operators needed because there are no cross-operator mutations
        g0 = Gene(0, '-1', target_historical_marking='-3073', new_connection_weight=1)
        g1 = Gene(1, '-1', target_historical_marking='-3073')
        g2 = Gene(1, '-1', target_historical_marking='3')
        g3 = Gene(1, '3', target_historical_marking='32')
        g4 = Gene(0, '-2', target_historical_marking='-3074', new_connection_weight=1)
        g5 = Gene(1, '-2', target_historical_marking='-3074')
        g6 = Gene(0, '3', target_historical_marking='32', new_connection_weight=1)
        
        a = Agent([g0, g1, g2, g3, g4, g5, g6], [X_SIZE * Y_SIZE * Z_SIZE, OUTPUT_NUM])
        '''
        a = Agent(genes, [32*32*3, 10])
        agents.append(a)
    return agents

# determine number of offspring according to rank
def getOffspringNum(rank):
    if rank <= INITIAL_POP_SIZE * .1:
        return 3
    if rank <= INITIAL_POP_SIZE * .2:
        return 2
    if rank <= INITIAL_POP_SIZE * .7:
        return 1
    return 0
    

print('\t>>\tWARNING: Not deleting outgoing references during hatch!')

generations = 10
train_x, train_y, _ = getCifarBatch(1)
validation_x, validation_y, _ = getCifarBatch(1)


agents = spawnMinimal(INITIAL_POP_SIZE)
for i_generation in range(generations):
    score_list = []
    # train and validate
    for a in agents:
        # train with data batch
        print('-----------------------------------------------------')
        print('>\tAgent Genes Info')
        #'''
        for gene in a.genes:
            gene.printInfo()
            #'''
        print(len(a.genes))
        for i_batch in range(0, len(train_y), STD_BATCH_SIZE):
            # BP
            #try:
            a.backPropagate(train_x[i_batch : (i_batch + STD_BATCH_SIZE)], \
                            train_y[i_batch : (i_batch + STD_BATCH_SIZE)])
            #except:
                #print('ERROR: The host of the above genes generated a no-gradient error during BP')
            #print(a.forward(data_x[i:(i+STD_BATCH_SIZE)]))
        # get accuracy on validation set
        score = a.getAccuracy(validation_x, validation_y)
        score_list.append(score)
    ranked_indices = np.argsort(np.array(score_list))  # rank according to accuracy
    # print champion performance
    max_score = score_list[ranked_indices[0]]
    print('Generation', i_generation, 'champion scoring:', str(max_score*10) + ' %')
    # reproduce according to ranking
    next_generation = []
    for rank in range(len(ranked_indices)):
        i_rank = ranked_indices[rank]
        a = agents[i_rank]
        n_offspring = getOffspringNum(rank)
        for i_offspring in range(n_offspring):
            next_generation.append(a.getAnOffspring())
    # deal with inaccuracies due to integer division
    while len(next_generation) > STD_POP_SIZE:
        next_generation.pop(-1)     # kill worst individual if too many agents
    while len(next_generation) < STD_POP_SIZE:
        next_generation.append(agents[ranked_indices[0]]) # add champion if not enof agents
    # update agents list
    agents = next_generation[:]