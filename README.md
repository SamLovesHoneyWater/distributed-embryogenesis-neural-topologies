# DENT — Distributed Embryogenesis of Neural Topologies

An evolutionary algorithm that discovers neural network topologies through a biologically-inspired developmental process: **embryogenesis** (growing neurons from genetic instructions) + **synaptogenesis** (forming connections based on spatial proximity) + **Lamarckian evolution** (backpropagation during fitness evaluation).

## Key Results

**100% accuracy** on the binocular retina task (Clune et al., 2013) — a benchmark specifically designed to test whether evolution can discover modular network architectures.

| Version | Best Accuracy | Key Innovation |
|---------|:------------:|----------------|
| v2 | 81.25% | Pure DENT (linear ceiling) |
| v3 | 87.89% | + Synaptogenesis |
| v3-deep | **100.0%** | + Deep initial topology |
| Reference MLP (2-layer) | 85% | Standard backprop only |
| Reference MLP (3-layer) | 98.4% | Standard backprop only |

## How It Works

### 1. Embryogenesis (Topology Discovery)
Each agent carries a **genome** of developmental genes. These genes operate on neurons in a 3D spatial grid:
- **AddConnection**: Grow a connection to a neuron with matching chemical markers
- **InsertNode**: Insert a new neuron on an existing connection

Genes activate based on **chemical markers** (histological markings) — neurons with matching markers are affected by the gene. This creates a developmental program where the same gene can affect multiple neurons simultaneously.

### 2. Synaptogenesis (Connection Growth)
After embryogenesis creates the topology, a second phase mimics biological **axon guidance**: hidden neurons grow additional connections to nearby input neurons based on **spatial proximity**. This is critical because:
- InsertNode creates neurons with only 1 input (linear pass-throughs)
- Synaptogenesis gives them multiple inputs (enabling non-linear computation)
- Spatial proximity naturally encourages **modularity** (left inputs connect to left hidden, right to right)

### 3. Lamarckian Learning
Each agent is trained with backpropagation during fitness evaluation. The learned weights are **not** inherited (only topology is inherited), but the training reveals whether a topology CAN learn the task. This is the Lamarckian principle: acquired traits (learned weights) influence fitness (survival), even though only the genetic topology is passed on.

### 4. Evolution
Tournament selection with elitism. Mutations include:
- Gene duplication and deletion
- Weight perturbation
- InsertNode (adding depth)
- Marker mutation (changing which neurons a gene affects)

## Files

- `dent_v3.py` — Core library: Gene, Node, Agent, speciation, modularity
- `evolve_deep.py` — Deep topology evolution (achieves 100%)
- `dent_v2.py` — Earlier version without synaptogenesis

## Running

```bash
# Set up environment
python3 -m venv dent-env
source dent-env/bin/activate
pip install numpy

# Run the evolution (achieves 100% on retina task)
python evolve_deep.py

# With custom parameters
python evolve_deep.py --generations 40 --pop-size 50
```

## The Retina Task

The binocular retina task (from Clune, Mouret & Lipson 2013) is specifically designed to test for **modularity** in evolved networks:

- 8 binary inputs: 4 for left visual field, 4 for right visual field
- Output: 1 if BOTH fields match their respective patterns, 0 otherwise
- The optimal solution processes left and right fields SEPARATELY, then combines
- This requires the network to discover modular structure

A non-modular network struggles because it must learn the XOR-like interaction between fields without any structural support. A modular network — one that separates left and right processing — can solve it easily.

## Biological Inspiration

| Biological Process | DENT Implementation |
|---|---|
| Genetic instructions | Gene objects with chemical markers |
| Neural development | Embryogenesis loop (gene activation) |
| Axon guidance | Synaptogenesis (spatial proximity) |
| Brain regions | Chemical markers group neurons |
| Synaptic plasticity | Backpropagation during evaluation |
| Natural selection | Tournament selection on fitness |
| Mutation | Gene duplication, deletion, modification |

## Key Insight: The Connectivity Bottleneck

The breakthrough from v2 (81%) to v3 (100%) came from identifying that InsertNode genes create hidden neurons with **only one source** — effectively linear pass-throughs. Adding synaptogenesis (spatial connection growth) gives hidden neurons multiple inputs, enabling the non-linear computation needed for complex tasks.

This mirrors real neural development: neurons are born, migrate to positions, and THEN grow connections to neighbors. The two-phase process (birth + wiring) is essential.
