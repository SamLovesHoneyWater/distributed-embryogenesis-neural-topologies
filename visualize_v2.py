#!/usr/bin/env python3
"""
Visualize DENT v2 evolved topologies.
Saves topology graphs as PNG files.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from dent_v2 import Agent, Gene, HyperParams, generate_retina_dataset, evolve


def visualize_agent(agent: Agent, filename: str = 'topology.png', 
                    title: str = ''):
    """Visualize an agent's network topology."""
    G = nx.DiGraph()
    
    node_colors = []
    node_labels = {}
    
    for i, node in enumerate(agent.nodes):
        if node.sources or node.outgoing:
            G.add_node(i)
            if node.node_type == 'input':
                node_colors.append('#4CAF50')  # Green
                node_labels[i] = f'I{node.io_index}'
            elif node.node_type == 'output':
                node_colors.append('#F44336')  # Red
                node_labels[i] = f'O{node.io_index}'
            else:
                node_colors.append('#9E9E9E')  # Gray
                node_labels[i] = f'H{i}'
    
    # Add edges
    for i, node in enumerate(agent.nodes):
        if i not in G.nodes:
            continue
        for src in node.sources:
            src_idx = agent.nodes.index(src)
            if src_idx in G.nodes:
                G.add_edge(src_idx, i)
    
    if len(G.nodes) == 0:
        print(f"No connected nodes to visualize for {filename}")
        return
    
    plt.figure(figsize=(12, 8))
    
    try:
        layout = nx.nx_pydot.graphviz_layout(G, prog='dot')
    except:
        layout = nx.spring_layout(G, k=2, iterations=50)
    
    # Reorder colors to match graph nodes
    colors = []
    labels = {}
    for n in G.nodes:
        idx = list(G.nodes).index(n)
        colors.append(node_colors[idx])
        labels[n] = node_labels.get(n, str(n))
    
    nx.draw(G, pos=layout, node_color=colors, with_labels=True,
            labels=labels, node_size=500, font_size=8,
            edge_color='#666666', arrows=True, arrowsize=10)
    
    info = (f"Nodes: {agent.n_nodes} | Connections: {agent.n_connections} | "
            f"Genes: {len(agent.genes)}")
    plt.title(f"{title}\n{info}", fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")


def main():
    os.makedirs('evolved_topologies_v2', exist_ok=True)
    
    print("Running evolution (50 generations, pop 40)...")
    pop, best_hist, mean_hist = evolve(
        generations=50, pop_size=40, verbose=True
    )
    
    # Find best agent
    data_x, data_y = generate_retina_dataset()
    best_agent = max(pop, key=lambda a: a.get_fitness(data_x, data_y))
    
    # Visualize best topology
    visualize_agent(
        best_agent, 
        'evolved_topologies_v2/best_topology.png',
        f'Best Evolved Topology (acc={best_agent.get_accuracy(data_x, data_y):.3f})'
    )
    
    # Plot fitness history
    plt.figure(figsize=(10, 5))
    plt.plot(best_hist, label='Best fitness', color='#2196F3')
    plt.plot(mean_hist, label='Mean fitness', color='#FF9800', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('DENT v2 — Fitness Over Generations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('evolved_topologies_v2/fitness_history.png', dpi=150)
    plt.close()
    print("Saved: evolved_topologies_v2/fitness_history.png")


if __name__ == '__main__':
    main()
