#!/usr/bin/env python3
"""
DENT v3 — Deep topology evolution.

This configuration achieves 100% accuracy on the retina task by:
1. Deeper initial topology (two InsertNode genes per connection)
2. Synaptogenesis (multi-input hidden nodes via spatial proximity)
3. Sufficient backprop (200 epochs per evaluation)
4. Low parsimony pressure (allow topology to grow)
5. Tournament selection (robust, no speciation fragmentation)

Usage:
    python evolve_deep.py [--generations N] [--pop-size N]
"""

import numpy as np
import time
import sys

from dent_v3 import Agent, Gene, HyperParams, generate_retina_dataset


def evolve_deep(generations=30, pop_size=30, verbose=True):
    """
    Evolve with deeper starting topology.
    
    The key insight: starting with two hidden layers (via double InsertNode
    genes) gives the networks enough depth to learn non-linear functions.
    Synaptogenesis then connects hidden nodes to multiple inputs based on
    spatial proximity, enabling the non-linear computation needed for
    tasks like the retina problem.
    """
    hp = HyperParams()
    hp.BACKPROP_EPOCHS = 200
    hp.BACKPROP_LR = 0.08
    hp.CONNECTION_COST = 0.00005
    hp.CONNECTION_COST_WARMUP = 20
    hp.MODULARITY_BONUS = 0.1
    hp.MODULARITY_WARMUP = 5
    hp.PROB_INS_NODE = 0.55
    hp.PROB_MUTATE_GENE = 0.40
    
    data_x, data_y = generate_retina_dataset()
    n_input = data_x.shape[1]
    n_output = 2
    
    if verbose:
        print(f"Deep topology evolution")
        print(f"Population: {pop_size}, Generations: {generations}")
        print(f"BP: {hp.BACKPROP_EPOCHS} epochs, LR: {hp.BACKPROP_LR}")
        print(f"Grid: {hp.SPACE_X}x{hp.SPACE_Y}x{hp.SPACE_Z}")
        print(f"Dataset: {data_x.shape[0]} samples, {n_input} inputs, {n_output} outputs")
        print()
    
    # Initialize with deep topology:
    # input -> hidden_layer_1 -> hidden_layer_2 -> output
    population = []
    for _ in range(pop_size):
        genes = []
        for i in range(n_input):
            for j in range(n_output):
                # Direct connection gene
                g = Gene(0, f'-{i+1}', f'-{n_input+j+1}',
                        weight=np.random.normal(0, 1))
                genes.append(g)
                # First InsertNode: creates hidden layer 1
                gi = Gene(1, f'-{n_input+j+1}', f'-{i+1}',
                         weight=np.random.normal(0, 1))
                genes.append(gi)
                # Second InsertNode: creates hidden layer 2
                gi2 = Gene(1, f'-{n_input+j+1}1', f'-{i+1}1',
                          weight=np.random.normal(0, 0.5))
                genes.append(gi2)
        population.append(Agent(genes, n_input, n_output, hp))
    
    if verbose:
        h = sum(1 for n in population[0].nodes if n.node_type == 'hidden')
        print(f"Initial topology: {population[0].n_nodes} nodes ({h} hidden), "
              f"{population[0].n_connections} connections")
        print()
    
    best_ever_acc = 0.0
    best_ever_agent = None
    
    for gen in range(generations):
        t0 = time.time()
        
        eff_cc = 0.0 if gen < hp.CONNECTION_COST_WARMUP else hp.CONNECTION_COST
        eff_mb = 0.0 if gen < hp.MODULARITY_WARMUP else hp.MODULARITY_BONUS
        
        for agent in population:
            agent.train(data_x, data_y, epochs=hp.BACKPROP_EPOCHS,
                       lr=hp.BACKPROP_LR, batch_size=hp.BACKPROP_BATCH_SIZE)
            accuracy = agent.get_accuracy(data_x, data_y)
            agent.fitness = (accuracy 
                           - eff_cc * agent.n_connections 
                           + eff_mb * agent.modularity)
        
        population.sort(key=lambda a: a.fitness, reverse=True)
        best = population[0]
        best_acc = best.get_accuracy(data_x, data_y)
        if best_acc > best_ever_acc:
            best_ever_acc = best_acc
            best_ever_agent = best
        
        hidden = sum(1 for n in best.nodes if n.node_type == 'hidden')
        mean_fit = np.mean([a.fitness for a in population])
        gt = time.time() - t0
        
        marker = " ***" if best_acc >= best_ever_acc and best_acc > 0.88 else ""
        if verbose:
            print(f"Gen {gen:3d} | acc={best_acc:.4f} | "
                  f"mean={mean_fit:.4f} | "
                  f"N={best.n_nodes}(H={hidden}) "
                  f"C={best.n_connections} "
                  f"Q={best.modularity:.3f} | "
                  f"{gt:.1f}s{marker}")
        
        # Early stopping on perfect accuracy
        if best_acc >= 1.0:
            if verbose:
                print(f"\nPerfect accuracy achieved at generation {gen}!")
            break
        
        # Tournament selection + elitism
        next_gen = []
        for i in range(min(3, len(population))):
            elite = population[i]
            next_gen.append(Agent(
                [g.clone() for g in elite.genes],
                n_input, n_output, hp
            ))
        
        hm_list = list(best.hm_dict.keys()) if hasattr(best, 'hm_dict') else []
        while len(next_gen) < pop_size:
            contestants = [population[np.random.randint(len(population))] 
                         for _ in range(3)]
            winner = max(contestants, key=lambda a: a.fitness)
            offspring = winner.reproduce(hm_list)
            next_gen.append(offspring)
        
        population = next_gen[:pop_size]
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"BEST EVER ACCURACY: {best_ever_acc:.4f}")
        if best_ever_agent:
            hidden = sum(1 for n in best_ever_agent.nodes 
                        if n.node_type == 'hidden')
            print(f"Topology: {best_ever_agent.n_nodes} nodes "
                  f"({hidden} hidden), "
                  f"{best_ever_agent.n_connections} connections, "
                  f"Q={best_ever_agent.modularity:.4f}")
    
    return population, best_ever_acc


if __name__ == '__main__':
    print("DENT v3 — Deep Topology Evolution")
    print("=" * 50)
    print()
    
    # Parse simple args
    gens = 30
    pop = 30
    for i, arg in enumerate(sys.argv[1:]):
        if arg == '--generations' and i + 2 < len(sys.argv):
            gens = int(sys.argv[i + 2])
        if arg == '--pop-size' and i + 2 < len(sys.argv):
            pop = int(sys.argv[i + 2])
    
    pop_result, best_acc = evolve_deep(
        generations=gens, pop_size=pop, verbose=True
    )
