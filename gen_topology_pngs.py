#!/usr/bin/env python3
"""
Generate evolved topology visualizations for the v3 system.
Runs evolution and saves network graph PNGs at key generations.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from collections import defaultdict
import time

from dent_v3 import Agent, Gene, HyperParams, generate_retina_dataset


def draw_topology(agent, ax, data_x=None, data_y=None):
    """Draw a detailed network topology with layered layout."""
    nodes = agent.nodes
    hp = agent.hp
    
    input_nodes = sorted([n for n in nodes if n.node_type == 'input'],
                         key=lambda n: n.io_index)
    hidden_nodes = sorted([n for n in nodes if n.node_type == 'hidden'],
                          key=lambda n: (n.coords[2], n.coords[0]))
    output_nodes = sorted([n for n in nodes if n.node_type == 'output'],
                          key=lambda n: n.io_index)
    
    n_in = len(input_nodes)
    half_in = n_in // 2
    mid_x_grid = hp.SPACE_X // 2
    
    pos = {}
    
    # Inputs: left retina on left, right retina on right
    total_width = 12.0
    gap = 2.0
    left_width = (total_width - gap) / 2
    
    for i, node in enumerate(input_nodes):
        if i < half_in:
            x = (i / max(1, half_in - 1)) * left_width if half_in > 1 else left_width / 2
        else:
            ri = i - half_in
            x = left_width + gap + (ri / max(1, half_in - 1)) * left_width if half_in > 1 else left_width + gap + left_width / 2
        pos[id(node)] = (x, 0)
    
    # Hidden: group by z-level, spread horizontally
    if hidden_nodes:
        z_levels = defaultdict(list)
        for n in hidden_nodes:
            z_levels[n.coords[2]].append(n)
        
        sorted_zs = sorted(z_levels.keys())
        n_z = len(sorted_zs)
        
        for li, z in enumerate(sorted_zs):
            y = 1.5 + (li + 1) * (4.0 / (n_z + 1))
            level = z_levels[z]
            
            # Sort by x-coordinate for spatial consistency
            level.sort(key=lambda n: n.coords[0])
            
            n_level = len(level)
            spread = min(total_width, max(n_level * 0.6, 4.0))
            center = total_width / 2
            
            for ni, node in enumerate(level):
                if n_level > 1:
                    x = center - spread/2 + ni * spread / (n_level - 1)
                else:
                    x = center
                pos[id(node)] = (x, y)
    
    # Outputs: centered at top
    for i, node in enumerate(output_nodes):
        x = total_width / 2 - 0.8 + i * 1.6
        pos[id(node)] = (x, 6.5)
    
    # Draw edges with weight-based coloring
    for node in nodes:
        if id(node) not in pos:
            continue
        x2, y2 = pos[id(node)]
        for si, src in enumerate(node.sources):
            if id(src) not in pos:
                continue
            x1, y1 = pos[id(src)]
            w = node.weights[si] if si < len(node.weights) else 0
            
            alpha = min(0.7, max(0.05, abs(w) * 0.2))
            width = max(0.2, min(2.5, abs(w) * 0.4))
            
            color = (0.15, 0.35, 0.75, alpha) if w > 0 else (0.75, 0.15, 0.15, alpha)
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)
    
    # Draw nodes
    node_size_input = 140
    node_size_hidden = 50
    node_size_output = 180
    
    # Inputs
    for i, node in enumerate(input_nodes):
        x, y = pos[id(node)]
        color = '#2ECC71' if i < half_in else '#E74C3C'
        ax.scatter(x, y, s=node_size_input, c=color, edgecolors='#2C3E50',
                   linewidth=1.0, zorder=5)
        label = f'L{i}' if i < half_in else f'R{i-half_in}'
        ax.annotate(label, (x, y), fontsize=5, ha='center', va='center',
                    zorder=6, fontweight='bold', color='white')
    
    # Hidden
    for node in hidden_nodes:
        if id(node) not in pos:
            continue
        x, y = pos[id(node)]
        if node.coords[0] < mid_x_grid:
            color = '#A8E6CF'
        else:
            color = '#FFB3BA'
        
        # Size proportional to number of connections
        n_conn = len(node.sources) + len(node.outgoing)
        size = max(25, min(80, 20 + n_conn * 5))
        
        ax.scatter(x, y, s=size, c=color, edgecolors='#7F8C8D',
                   linewidth=0.5, zorder=5)
    
    # Outputs
    for node in output_nodes:
        x, y = pos[id(node)]
        ax.scatter(x, y, s=node_size_output, c='#F1C40F', edgecolors='#2C3E50',
                   linewidth=1.5, zorder=5, marker='s')
        ax.annotate(f'O{node.io_index}', (x, y), fontsize=7, ha='center',
                    va='center', zorder=6, fontweight='bold')
    
    ax.set_xlim(-0.5, total_width + 0.5)
    ax.set_ylim(-0.8, 7.3)
    ax.set_aspect('equal')
    ax.axis('off')


def main():
    data_x, data_y = generate_retina_dataset()
    n_input = data_x.shape[1]
    n_output = 2
    
    hp = HyperParams()
    hp.BACKPROP_EPOCHS = 150
    hp.BACKPROP_LR = 0.08
    hp.CONNECTION_COST = 0.00005
    hp.CONNECTION_COST_WARMUP = 15
    hp.MODULARITY_BONUS = 0.1
    hp.MODULARITY_WARMUP = 3
    hp.PROB_INS_NODE = 0.55
    hp.PROB_MUTATE_GENE = 0.40
    
    pop_size = 30
    generations = 25
    
    # Generations to snapshot
    snapshot_gens = {0, 3, 6, 10, 15, 20, 24}
    
    np.random.seed(42)
    
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
    
    outdir = 'evolved_topologies/v3_synaptogenesis'
    os.makedirs(outdir, exist_ok=True)
    
    print(f"Running evolution (pop={pop_size}, gen={generations})...")
    print(f"Will snapshot at generations: {sorted(snapshot_gens)}")
    
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
        gt = time.time() - t0
        
        marker = " ***" if best_acc >= best_ever_acc and best_acc > 0.85 else ""
        print(f"  Gen {gen:3d}: acc={best_acc:.4f} H={hidden} C={best.n_connections} "
              f"Q={best.modularity:.3f} | {gt:.1f}s{marker}")
        
        if gen in snapshot_gens:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            fig.patch.set_facecolor('white')
            
            draw_topology(best, ax, data_x, data_y)
            
            acc_pct = f'{best_acc:.1%}'
            title = (f'Generation {gen}  —  Accuracy: {acc_pct}\n'
                     f'{best.n_nodes} nodes ({hidden} hidden)  •  '
                     f'{best.n_connections} connections  •  Q={best.modularity:.3f}')
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
            
            # Legend
            legend_elements = [
                mpatches.Patch(facecolor='#2ECC71', edgecolor='#2C3E50', label='Left retina'),
                mpatches.Patch(facecolor='#E74C3C', edgecolor='#2C3E50', label='Right retina'),
                mpatches.Patch(facecolor='#A8E6CF', edgecolor='#7F8C8D', label='Hidden (left-side)'),
                mpatches.Patch(facecolor='#FFB3BA', edgecolor='#7F8C8D', label='Hidden (right-side)'),
                mpatches.Patch(facecolor='#F1C40F', edgecolor='#2C3E50', label='Output'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=7,
                      framealpha=0.9)
            
            path = os.path.join(outdir, f'G{gen:02d}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"    → Saved {path}")
        
        # Reproduce
        next_gen = []
        for i in range(3):
            next_gen.append(Agent([g.clone() for g in population[i].genes], n_input, n_output, hp))
        hm_list = list(best.hm_dict.keys()) if hasattr(best, 'hm_dict') else []
        while len(next_gen) < pop_size:
            c = [population[np.random.randint(len(population))] for _ in range(3)]
            next_gen.append(max(c, key=lambda a: a.fitness).reproduce(hm_list))
        population = next_gen[:pop_size]
    
    print(f"\nBest ever accuracy: {best_ever_acc:.4f}")
    print(f"Topologies saved to {outdir}/")


if __name__ == '__main__':
    main()
