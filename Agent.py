# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:07:12 2021

@author: Sammy
"""

import tensorflow as tf
from Node import Node
from Gene import Gene
from Include import *

class Agent():
    def __init__(self, genes, io_pair):
        #self.debug_counter = 0
        
        self.genes = genes
        self.n_input = io_pair[0]   # number of input nodes
        self.n_output = io_pair[1]   # number of output nodes
        self.lr = LEARNING_RATE
        self.decay = LEARNING_RATE_DECAY
        
        self.nn = []    # network topology
        # spatial representation of network for easy reference
        self.grid = np.array([[[None for x in range(SPACE_X)] for y in range(SPACE_Y)] for z in range(SPACE_Z)])
        self.hm_dict = {}
        ##########################################################
        self.hm_grid = np.zeros([SPACE_Z, SPACE_Y, SPACE_X], dtype = 'int32')
        ##########################################################
        
        self.AE()       # generate topology from genes
        #t0 = time.time()
        self.makeNet()   # create functional NN from topology
        #print('>\tTotal makeNet time:',time.time()-t0)
        
    # returns the node at a coordinate
    def getNodeAt(self, coords):
        x, y, z = coords
        return self.grid[z][y][x]
    
    # changes the historical marking of a node and updates the hm grid
    def addMarking(self, node, op):
        # i/o nodes have static negative markings
        if node.node_type != 'hidden':
            return
        
        hm = node.markings
        same_hm_nodes = self.hm_dict[hm]
        
        # if other nodes share own marking, delete reference to self only
        if len(same_hm_nodes) > 1:
            same_hm_nodes.pop(same_hm_nodes.index(node))
        # otherwise, no node has own marking, delete the key entirely
        else:
            del self.hm_dict[hm]
        # update node attributes
        m = str(op)
        node.markings += m
        # create new reference
        new_hm = node.markings
        if new_hm not in self.hm_dict.keys():
            self.hm_dict[new_hm] = []
        self.hm_dict[new_hm].append(node)
        ##########################################################
        x, y, z = node.coords
        self.hm_grid[z][y][x] = eval(new_hm)
        ##########################################################

    # adds the node to the NN list and references
    def writeNode(self, node):
        self.nn.append(node)    # write to overall list
        x, y, z = node.coords
        # write to node grid, cautioning overwrite
        if self.grid[z][y][x] is not None:
            result = 0
            print('Warning: Overwriting existing node')
        else:
            result = 1
        # update references
        self.grid[z][y][x] = node
        hm = node.markings
        if hm not in self.hm_dict.keys():
            self.hm_dict[hm] = []
        self.hm_dict[hm].append(node)
        ##########################################################
        self.hm_grid[z][y][x] = eval(hm)
        ##########################################################
        return result
    
    # start artifitial embryogeny
    def AE(self):
        self.spawnMinimalNet()  # create minimal nn
        active_genes = self.genes[:]    # genes that are not yet exhausted
        activity_list = [gene.exp_count for gene in active_genes]
        moving = True
        t0 = time.time()
        # decode genes to get the topology of phenotype NN
        while len(active_genes) != 0 and moving:
            if time.time() - t0 > MAX_EMBRYO_TIME:
                self.raiseAETimeout()
                break
            moving = False
            for gene_index in range(-len(active_genes), 0):    # negative indexing becoz of popping
                successful = False
                gene = active_genes[gene_index]
                if gene.own_hm not in self.hm_dict.keys():
                    continue    # skip if there are no applicable genes
                for node in self.hm_dict[gene.own_hm]:
                    result = self.applyGene(node, gene)
                    if (not successful) and result:
                        successful = True
                    if (not gene.ever_activated) and result:
                        gene.ever_activated = True
                        ''' 
                if False:#(time.time()-t0) > 0.01:
                    debug_counter += 1
                    print(time.time()-t0)
                    print('operator:',gene.op)
                    print('gene index:',gene_index)
                    print('source:',gene.own_hm)
                    print('taret:',gene.target_hm)
                    #print('all source node coords:')
                    print()
                t0 = time.time()
                '''
                        
                # successful application
                if successful:
                    activity_list[gene_index] -= 1
                    # deactivate exhausted gene
                    if activity_list[gene_index] <= 0:
                        active_genes.pop(gene_index)
                        activity_list.pop(gene_index)
                        #print('number of left genes:', len(active_genes))
                    
                if (not moving) and successful:
                    moving = True
        #print('>\ttotal AE time:', time.time()-t0)
        
    def raiseAETimeout(self):
        print('\t>>\tWARNING: Embryogeny timeout. See agent gene info below:')
        print('-----------------------------------------------------')
        for gene in self.genes:
            gene.printInfo()
        print('-----------------------------------------------------')
        
    # fixate NN structure and pass params for future training
    def makeNet(self):
        for node in self.nn:
            node.hatch()
        # clear reference variables
        del self.grid
        # del self.hm_dict
        del self.hm_grid
        
    # feed forward
    def forward(self, x):
        n_data = x.shape[0]
        x = tf.Variable(x, dtype = STD_DTYPE)
        for node in self.nn:
            if node.node_type == 'input':
                # data as the activation of input nodes
                node.a = x[:, node.io_index]
                node.fed = True
            else:
                node.ff1()
        
        # register outputs
        y_list = []
        fake_output = 0
        for node in self.output_nodes:
            if not node.fed:
                fake_output += 1
                y_list.append(tf.Variable([0 for i in range(n_data)], dtype = STD_DTYPE))
            else:
                y_list.append(node.a)
        
        # if none of the outputs come from actual feed forward, this is a stupid net
        if fake_output == self.n_output:
            return False
        
        # reset fed status for all nodes
        self.resetNet()
        if FINAL_SOFTMAX_MASK:
            y_flat = tf.concat(y_list, axis = 0)
            y_flipped = tf.reshape(y_flat, (self.n_output, n_data))
            y = tf.transpose(y_flipped)
            a = tf.nn.softmax(y)
            return a
        return y_list
    
    # network loss
    def getLoss(self, y, y_):
        if FINAL_SOFTMAX_MASK:
            return tf.compat.v1.losses.softmax_cross_entropy(y_, y)
        if OUTPUT_LAYER_ACTIVATION == 'sigmoid':
            return tf.compat.v1.losses.sigmoid_cross_entropy(y_, y)            
        
    # learn params
    def backPropagate(self, x, y_):
        y_ = tf.Variable(choice2onehot(y_), dtype = STD_DTYPE)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = self.lr)
        with tf.GradientTape() as tape:
            y = self.forward(x)
            if y is False:
                return
            loss = self.getLoss(y, y_)
        grads = tape.gradient(loss, [node.w for node in self.nn])
        #print("Gradients:", grads)
        #print("Loss:", loss)
        optimizer.apply_gradients(zip(grads, [node.w for node in self.nn]), \
                                  global_step = tf.compat.v1.train.get_or_create_global_step())

        self.lr *= self.decay
    
    def getAccuracy(self, data_x, data_y, in_percentage = False):
        
        try:
            y = self.forward(data_x)
        except:
            self.visualizeConns()
            print(ascno)
        if y is False:
            return 0.0
        y = tf.argmax(y, axis = 1)
        y = y.numpy()
        correct_list = np.equal(y, data_y)
        correct = correct_list.sum()
        
        '''
        ###################################
        print('pooling')
        predictions = pool.map(self.forward, data_x)
        for y, y_ in zip(predictions, data_y):
            correct += (int(tf.argmax(y)) == y_)  # count if is correct
        ###################################
        
        ###################################
        correct = 0
        for x, y_ in zip(data_x, data_y):
            y = self.forward(x)
            #print(int(tf.argmax(y)), y_)
            correct += (int(tf.argmax(y)) == y_)  # count if is correct
        ###################################
        '''
        
        accuracy = correct / len(data_y)
        if in_percentage:
            accuracy *= 100
        return accuracy
    
    # reset network after each ff & bp
    def resetNet(self):
        for node in self.nn:
            node.fed = False
    
    # spawn minimal network (fully connected, no hidden layer)
    def spawnMinimalNet(self):        
        # add input nodes
        for i in range(self.n_input):
            conns = [[], []]     # input nodes have no initial conns
            coords = getInputNodePos(i)
            node = Node(conns, coords, node_type='input', activation='linear',\
                        io_index = i)
            node.markings += '-1'   # uniform input node hm
            self.writeNode(node)
        
        # add output nodes
        self.output_nodes = []
        for i in range(self.n_output):
            conns = [[], []]     # output nodes have no initial conns
            coords = getOutputNodePos(i)
            node = Node(conns, coords, node_type='output', activation=OUTPUT_LAYER_ACTIVATION,\
                        io_index = i)
            node.markings += '-2'  # uniform output node hm            
            self.writeNode(node)
            self.output_nodes.append(node)  # maintain an output node list for reference during ff
        return True        # successful spawn
    
    def applyGene(self, node, gene):
        # add connection operation
        if gene.op == 0:
            return self.addConn(node, gene)
        # insert node operation
        if gene.op == 1:
            return self.insNode(node, gene)
        # application failed
        return False
    
    def addConn(self, node0, gene):
        hm = gene.target_hm
        candidates = np.where(self.hm_grid == eval(hm))
        zs, ys, xs = candidates
        if len(xs):   
            x0, y0, z0 = node0.coords
            mask = np.zeros_like(xs)
            mask.fill(x0)
            x2 = np.square(xs - mask)
            mask.fill(y0)
            y2 = np.square(ys - mask)
            mask.fill(z0)
            z2 = np.square(zs - mask)
            distances = x2 + y2 + z2
            i_list = np.argsort(distances)
            for i in i_list:
                coords1 = [xs[i], ys[i], zs[i]]
                node1 = self.getNodeAt(coords1)
                # no existing connection or self-connection
                if (node1 in node0.outgoing) or (node0 == node1):
                    continue    # check next candidate
                # too far away
                if distances[i] > MAX_CONN_LEN:
                    break   # no need to check proceeding candidates, they are farther
                # no cyclic connection
                if node1.isFatherOf(node0):
                    continue    # check next candidate
                node0.outgoing.append(node1)
                node1.sources.append(node0)
                node1.w.append(gene.w)
                # update historical markings
                self.addMarking(node0, 1)
                self.addMarking(node1, 1)
                return True
            
        # didn't find unrepetitive target
        return False

    def insNode(self, node2, gene):
        hm = gene.target_hm
        for i in range(len(node2.w)):
            node1 = node2.sources[i]
            w = node2.w[i]
            if node1.markings == hm:
                # place new node on the grid
                candidates = np.where((self.hm_grid == 0))
                zs, ys, xs = candidates
                if len(xs):
                    # get middle coords
                    mid_coords = [0, 0, 0]
                    for dim in [0, 1, 2]:
                        mid_coords[dim] = (node1.coords[dim] + node2.coords[dim])//2
                    x0, y0, z0 = mid_coords
                    mask = np.zeros_like(xs)
                    mask.fill(x0)
                    x2 = np.square(xs - mask)
                    mask.fill(y0)
                    y2 = np.square(ys - mask)
                    mask.fill(z0)
                    z2 = np.square(zs - mask)
                    distances = x2 + y2 + z2
                    nearest_i = np.argsort(distances)[0]      # ranked list of indices
                    # all good, insert node
                    new_coords = [xs[nearest_i], ys[nearest_i], zs[nearest_i]]
                    conn = [[[node1, w]], [node2]]
                    new_node = Node(conn, new_coords)
                    new_node.markings = '3'
                    self.writeNode(new_node)
                    node1.outgoing[node1.outgoing.index(node2)] = new_node
                    node2.sources[i] = new_node
                    node2.w[i] = gene.w
                    # update historical markings
                    self.addMarking(node1, 2)
                    self.addMarking(node2, 2)
                    return True
                
        # didn't successfully apply operator
        return False
    
    def getAnOffspring(self):
        new_genes = []
        hm_list = list(self.hm_dict.keys())
        
        # self mutations
        for g in self.genes:
            # trash genes are not passed on
            if not g.ever_activated:
                continue
            # replicative gene
            if random.random() < PROB_REPLICATE_SELF:
                new_gene = g.getClone()
                new_genes.append(new_gene)
            
            # delete gene
            random_number = random.random()
            if random_number < PROB_DEL_SELF:
                continue
            random_number -= PROB_DEL_SELF

            # mutate gene
            if random_number < PROB_CHANGE_SELF:
                r = random.random()            
                # self mutations are mutually exclusive
                # weight mutations and insert-node mutations come first
                # so that op1 genes have even probs for other mutations
                if g.op == 0 and r < PROB_CHANGE_WEIGHT:
                    new_genes.append(g.mutateWeight())
                    continue
                r -= PROB_CHANGE_WEIGHT
                
                # insert-node mutation
                if g.op == 0 and r < PROB_INS_NODE:
                    new_genes.append(g.getClone())
                    new_genes.append(g.getInsGene())
                    continue
                r -= PROB_INS_NODE
                
                # changes initial uses-before-expiration
                if r < PROB_CHANGE_ACTIVITY:
                    new_genes.append(g.mutateActivity())
                    continue
                r -= PROB_CHANGE_ACTIVITY
                
                if r < PROB_CHANGE_OWN_HM:
                    new_g = g.getClone()
                    i_marking = random.randint(0, len(hm_list)-1)
                    # source should not be an output node
                    while i_marking == 1:
                        i_marking = random.randint(0, len(hm_list)-1)
                    new_g.mutateOwnHM(hm_list[i_marking])
                    new_genes.append(new_g)
                    continue
                r -= PROB_CHANGE_OWN_HM
                
                if r < PROB_CHANGE_TARGET_HM:
                    new_g = g.getClone()
                    # outgoing connection should not point towards an input node
                    i_marking = random.randint(1, len(hm_list)-1)
                    new_g.mutateTargetHM(hm_list[i_marking])
                    new_genes.append(new_g)
                    continue
                r -= PROB_CHANGE_TARGET_HM
                
            # non-mutative inheritance
            new_gene = g.getClone()
            new_genes.append(new_gene)
        
        # add new random op0 gene at the end of genome
        if random.random() < PROB_NEW_RANDOM_CONN:
            source_hm_index = random.randint(0, len(hm_list)-1)
            # source should not be an output node
            while source_hm_index == 1:
                source_hm_index = random.randint(0, len(hm_list)-1)
            source = hm_list[source_hm_index]
            # target should not be an input node
            target_hm_index = random.randint(1, len(hm_list)-1)
            target = hm_list[target_hm_index]
            new_gene = Gene(0, source, target_historical_marking = target,\
                            new_connection_weight=0)
            # instead of w=0, randomly initialize weight
            new_gene = new_gene.mutateWeight()
            new_genes.append(new_gene)
        
        offspring = Agent(new_genes, [X_SIZE * Y_SIZE * Z_SIZE, OUTPUT_NUM])
        return offspring
    
    def visualizeConns(self, generation = None):
        G = nx.DiGraph()
        #hidden_iterator = 0
        for node in self.nn:
            if node.sources != [] or node.outgoing != []:
                G.add_node(self.nn.index(node))
                '''
                if node.node_type != 'hidden':
                    G.add_node(node.markings)
                else:
                    G.add_node('h_' + str(hidden_iterator))
                    hidden_iterator += 1
                    '''
        for node2 in self.nn:
            if node2.sources != []:
                for node1 in node2.sources:
                    G.add_edge(self.nn.index(node1), self.nn.index(node2))
        
        color_map = []
        for node in G:
            if node < self.n_input:
                color_map.append('green')
            elif node < self.n_input + self.n_output:
                color_map.append('red')
            else:
                color_map.append('gray')


        
        # clear previous plots, draw, and show
        plt.clf()
        layout = nx.nx_pydot.graphviz_layout(G, prog='dot')
        #layout = nx.drawing.nx_agraph.graphviz_layout(G)
        nx.draw(G, pos=layout, node_color=color_map, with_labels=True)
        if generation:
            plt.savefig('R' + str(HAPPY_NUMBER) + 'G'+str(generation)+'.png')
        plt.show()
        
                

if __name__ == '__main__':
    '''
    from multiprocessing import Pool
    from multiprocessing.dummy import Pool as ThreadPool
    pool = ThreadPool()
    print('threadpool not included in main code!!!!!')
    '''
    
    genes = []
    for i in range(1, 3072):
        g = Gene(0,str(-i), target_historical_marking = str(-32*32*3-(i-1)%10-1), new_connection_weight = 1)
        genes.append(g)
        g = Gene(1, str(-i), target_historical_marking = str(-32*32*3-(i-1)%10-1))
        genes.append(g)
    print('\n\n>\tInitializing Agent')
    a = Agent(genes, [32*32*3, 10])
    print('>\tInitialized Agent\n\n')
    #a.visualizeConns()
    data_x, data_y, _ = getCifarBatch(1)
    
    
    t0 = time.time()
    print(a.getAccuracy(data_x, data_y))
    print('10000 data forward time:',time.time()-t0)
    
    print(xkndofw)
    
    for i in range(0, 10000, STD_BATCH_SIZE):
        i = 0
        a.backPropagate(data_x[i:(i+STD_BATCH_SIZE)], data_y[i:(i+STD_BATCH_SIZE)])
        print(a.forward(data_x[i:(i+STD_BATCH_SIZE)]))
    '''
    for epoch in range(3):
        for i in range(10000):
            a.backPropagate(data_x[i], data_y[i])
            if i%100 == 0:
                y = a.forward(data_x[i])
                y_ = data_y[i]
                y_ = choice2onehot(y_)
                print(y, y_)
                '''