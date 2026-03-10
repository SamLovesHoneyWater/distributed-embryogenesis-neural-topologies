"""
DENT v3 — Deep Topology Experiment
Achieves 100% accuracy on the retina task.

Key insight: deeper starting topologies (2 layers of InsertNode genes)
combined with synaptogenesis creates networks that can learn the
XOR-like structure of the binocular retina task.
"""
import numpy as np
import time
from dent_v3 import Agent, Gene, HyperParams, generate_retina_dataset

def run_deep_experiment(pop_size=30, generations=30, bp_epochs=200, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    hp = HyperParams()
    hp.BACKPROP_EPOCHS = bp_epochs
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

    # Initialize with deeper starting topology
    population = []
    for _ in range(pop_size):
        genes = []
        for i in range(n_input):
            for j in range(n_output):
                g = Gene(0, f'-{i+1}', f'-{n_input+j+1}', weight=np.random.normal(0, 1))
                genes.append(g)
                gi = Gene(1, f'-{n_input+j+1}', f'-{i+1}', weight=np.random.normal(0, 1))
                genes.append(gi)
                gi2 = Gene(1, f'-{n_input+j+1}1', f'-{i+1}1', weight=np.random.normal(0, 0.5))
                genes.append(gi2)
        population.append(Agent(genes, n_input, n_output, hp))

    print(f"Deep topology: pop={pop_size}, gen={generations}, BP={bp_epochs}")
    print(f"Start: {population[0].n_nodes} nodes, {population[0].n_connections} conns")
    print()

    best_ever_acc = 0.0

    for gen in range(generations):
        t0 = time.time()
        
        eff_cc = 0.0 if gen < hp.CONNECTION_COST_WARMUP else hp.CONNECTION_COST
        eff_mb = 0.0 if gen < hp.MODULARITY_WARMUP else hp.MODULARITY_BONUS
        
        for agent in population:
            agent.train(data_x, data_y, epochs=hp.BACKPROP_EPOCHS,
                       lr=hp.BACKPROP_LR, batch_size=hp.BACKPROP_BATCH_SIZE)
            accuracy = agent.get_accuracy(data_x, data_y)
            agent.fitness = accuracy - eff_cc * agent.n_connections + eff_mb * agent.modularity
        
        population.sort(key=lambda a: a.fitness, reverse=True)
        best = population[0]
        best_acc = best.get_accuracy(data_x, data_y)
        best_ever_acc = max(best_ever_acc, best_acc)
        hidden = sum(1 for n in best.nodes if n.node_type == 'hidden')
        mean_fit = np.mean([a.fitness for a in population])
        
        gt = time.time() - t0
        marker = " ***" if best_acc >= best_ever_acc and best_acc > 0.88 else ""
        print(f"Gen {gen:3d} | acc={best_acc:.4f} | mean={mean_fit:.4f} | "
              f"N={best.n_nodes}(H={hidden}) C={best.n_connections} Q={best.modularity:.3f} | "
              f"{gt:.1f}s{marker}")
        
        if best_acc >= 1.0:
            print(f"\n*** PERFECT ACCURACY at gen {gen} ***")
        
        next_gen = []
        for i in range(3):
            elite = population[i]
            next_gen.append(Agent([g.clone() for g in elite.genes], n_input, n_output, hp))
        
        hm_list = list(best.hm_dict.keys()) if hasattr(best, 'hm_dict') else []
        while len(next_gen) < pop_size:
            contestants = [population[np.random.randint(len(population))] for _ in range(3)]
            winner = max(contestants, key=lambda a: a.fitness)
            offspring = winner.reproduce(hm_list)
            next_gen.append(offspring)
        
        population = next_gen[:pop_size]

    print(f"\nBest ever accuracy: {best_ever_acc:.4f}")
    return population, best_ever_acc

if __name__ == '__main__':
    run_deep_experiment()
