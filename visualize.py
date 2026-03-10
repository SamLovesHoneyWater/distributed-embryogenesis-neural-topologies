#!/usr/bin/env python3
"""
DENT Topology Visualizer — ASCII art visualization of evolved neural topologies.

Renders the network structure as a layered diagram showing:
- Input nodes (bottom), hidden nodes (middle), output nodes (top)
- Connection patterns between layers
- Spatial clustering (left/right retina modularity)
- Node connectivity statistics
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from dent_v3 import Agent, Gene, HyperParams, generate_retina_dataset


def visualize_topology(agent, title="Network Topology"):
    """ASCII visualization of an agent's neural topology."""
    nodes = agent.nodes
    
    input_nodes = [n for n in nodes if n.node_type == 'input']
    hidden_nodes = [n for n in nodes if n.node_type == 'hidden']
    output_nodes = [n for n in nodes if n.node_type == 'output']
    
    # Sort by x coordinate for spatial layout
    input_nodes.sort(key=lambda n: n.coords[0])
    hidden_nodes.sort(key=lambda n: (n.coords[2], n.coords[0]))  # z then x
    output_nodes.sort(key=lambda n: n.coords[0])
    
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Nodes: {len(nodes)} ({len(input_nodes)} in, {len(hidden_nodes)} hidden, {len(output_nodes)} out)")
    print(f"  Connections: {agent.n_connections}")
    print(f"  Modularity Q: {agent.modularity:.4f}")
    
    # Layer statistics
    print(f"\n  --- Input Layer ({len(input_nodes)} nodes) ---")
    half = len(input_nodes) // 2
    left_inputs = input_nodes[:half]
    right_inputs = input_nodes[half:]
    
    left_str = ' '.join(f'I{n.io_index}' for n in left_inputs)
    right_str = ' '.join(f'I{n.io_index}' for n in right_inputs)
    print(f"  Left retina:  [{left_str}]")
    print(f"  Right retina: [{right_str}]")
    
    # Hidden layer analysis — group by z-level for depth visualization
    if hidden_nodes:
        z_levels = {}
        for n in hidden_nodes:
            z = n.coords[2]
            if z not in z_levels:
                z_levels[z] = []
            z_levels[z].append(n)
        
        print(f"\n  --- Hidden Layers ({len(hidden_nodes)} nodes, {len(z_levels)} z-levels) ---")
        for z in sorted(z_levels.keys()):
            level_nodes = z_levels[z]
            # Classify by spatial position (left/right)
            mid_x = HyperParams.SPACE_X // 2
            left_h = [n for n in level_nodes if n.coords[0] < mid_x]
            right_h = [n for n in level_nodes if n.coords[0] >= mid_x]
            
            n_sources_avg = np.mean([len(n.sources) for n in level_nodes]) if level_nodes else 0
            n_out_avg = np.mean([len(n.outgoing) for n in level_nodes]) if level_nodes else 0
            
            print(f"  z={z:2d}: {len(level_nodes):2d} nodes "
                  f"(L={len(left_h)}, R={len(right_h)}) "
                  f"avg_in={n_sources_avg:.1f} avg_out={n_out_avg:.1f}")
    
    # Output layer
    print(f"\n  --- Output Layer ({len(output_nodes)} nodes) ---")
    for n in output_nodes:
        n_hidden_src = sum(1 for s in n.sources if s.node_type == 'hidden')
        n_input_src = sum(1 for s in n.sources if s.node_type == 'input')
        print(f"  O{n.io_index}: {len(n.sources)} sources "
              f"({n_input_src} from inputs, {n_hidden_src} from hidden)")
    
    # Connection pattern analysis
    print(f"\n  --- Connectivity Analysis ---")
    
    # How many hidden nodes connect to left vs right inputs?
    left_connected = 0
    right_connected = 0
    both_connected = 0
    
    left_input_ids = set(id(n) for n in left_inputs)
    right_input_ids = set(id(n) for n in right_inputs)
    
    for h in hidden_nodes:
        sources_ids = set(id(s) for s in h.sources)
        has_left = bool(sources_ids & left_input_ids)
        has_right = bool(sources_ids & right_input_ids)
        
        # Also check indirect connections (through other hidden nodes)
        queue = list(h.sources)
        visited = set()
        while queue:
            s = queue.pop(0)
            if id(s) in visited:
                continue
            visited.add(id(s))
            if id(s) in left_input_ids:
                has_left = True
            if id(s) in right_input_ids:
                has_right = True
            if s.node_type == 'hidden':
                queue.extend(s.sources)
        
        if has_left and has_right:
            both_connected += 1
        elif has_left:
            left_connected += 1
        elif has_right:
            right_connected += 1
    
    if hidden_nodes:
        print(f"  Hidden nodes connected to:")
        print(f"    Left retina only:  {left_connected} ({100*left_connected/len(hidden_nodes):.0f}%)")
        print(f"    Right retina only: {right_connected} ({100*right_connected/len(hidden_nodes):.0f}%)")
        print(f"    Both retinas:      {both_connected} ({100*both_connected/len(hidden_nodes):.0f}%)")
        
        modularity_indicator = "MODULAR" if (left_connected + right_connected) > both_connected else "INTEGRATED"
        print(f"  Structure: {modularity_indicator}")
    
    # Depth analysis
    print(f"\n  --- Depth Analysis ---")
    max_depth = 0
    for out in output_nodes:
        depth = _compute_depth(out)
        max_depth = max(max_depth, depth)
    print(f"  Maximum depth (input → output): {max_depth}")
    
    # Weight statistics
    all_weights = []
    for n in nodes:
        all_weights.extend(n.weights)
    if all_weights:
        w = np.array(all_weights)
        print(f"\n  --- Weight Statistics ---")
        print(f"  Mean: {np.mean(w):.3f}, Std: {np.std(w):.3f}")
        print(f"  Min: {np.min(w):.3f}, Max: {np.max(w):.3f}")
        print(f"  |w| > 1.0: {np.sum(np.abs(w) > 1.0)}/{len(w)} ({100*np.mean(np.abs(w) > 1.0):.0f}%)")
    
    print(f"{'='*60}")


def _compute_depth(node, visited=None):
    """Compute the depth from inputs to this node."""
    if visited is None:
        visited = set()
    
    if id(node) in visited:
        return 0
    visited.add(id(node))
    
    if not node.sources:
        return 0
    
    max_src_depth = 0
    for src in node.sources:
        d = _compute_depth(src, visited)
        max_src_depth = max(max_src_depth, d)
    
    return max_src_depth + 1


def run_and_visualize():
    """Run a short evolution and visualize the best topology."""
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
    pop_size = 20
    generations = 15  # Quick run for visualization
    
    print("Running evolution for topology visualization...")
    print(f"Pop: {pop_size}, Gen: {generations}")
    
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
    
    for gen in range(generations):
        for agent in population:
            agent.train(data_x, data_y, epochs=hp.BACKPROP_EPOCHS, 
                       lr=hp.BACKPROP_LR, batch_size=hp.BACKPROP_BATCH_SIZE)
            accuracy = agent.get_accuracy(data_x, data_y)
            eff_mb = 0.0 if gen < hp.MODULARITY_WARMUP else hp.MODULARITY_BONUS
            agent.fitness = accuracy + eff_mb * agent.modularity
        
        population.sort(key=lambda a: a.fitness, reverse=True)
        best = population[0]
        acc = best.get_accuracy(data_x, data_y)
        hidden = sum(1 for n in best.nodes if n.node_type == 'hidden')
        print(f"  Gen {gen}: acc={acc:.4f} N={best.n_nodes}(H={hidden}) C={best.n_connections}")
        
        next_gen = []
        for i in range(2):
            next_gen.append(Agent([g.clone() for g in population[i].genes], n_input, n_output, hp))
        hm_list = list(best.hm_dict.keys()) if hasattr(best, 'hm_dict') else []
        while len(next_gen) < pop_size:
            c = [population[np.random.randint(len(population))] for _ in range(3)]
            next_gen.append(max(c, key=lambda a: a.fitness).reproduce(hm_list))
        population = next_gen[:pop_size]
    
    # Visualize best
    best = max(population, key=lambda a: a.fitness)
    best.train(data_x, data_y, epochs=200, lr=0.08, batch_size=256)
    acc = best.get_accuracy(data_x, data_y)
    visualize_topology(best, f"Best Evolved Topology (acc={acc:.4f})")


if __name__ == '__main__':
    run_and_visualize()
