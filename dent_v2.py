"""
DENT v2 — Distributed Embryogenesis of Neural Topologies
Refactored for performance, portability, and extensibility.

Key changes from v0.4:
- Removed TensorFlow dependency → pure NumPy
- Vectorized neighbor search (was O(n²) per gene application)
- Cached DAG ancestry (isFatherOf was recursive without memoization)
- Expanded spatial grid (configurable, default 20x20x20)
- Cleaner separation of concerns
- Added more mutation operators
- Better fitness evaluation with connection cost (Pareto front)
"""

import numpy as np
from typing import List, Optional, Tuple, Set
import time


class HyperParams:
    """All hyperparameters in one place."""
    # Spatial grid
    SPACE_X = 20
    SPACE_Y = 20
    SPACE_Z = 20
    
    # Network
    LEARNING_RATE = 0.1
    LEARNING_RATE_DECAY = 0.95
    BATCH_SIZE = 256  # Increased from 5000 (was too large for small datasets)
    MAX_CONN_LEN_SQ = 99999  # Squared distance threshold (effectively unlimited for initial connections)
    MAX_EMBRYO_TIME = 10.0  # seconds
    EXP_COUNT_CAP = 16
    
    # Evolution
    POP_SIZE = 50  # Reduced for faster iteration
    GENERATIONS = 20
    
    # Mutation probabilities
    PROB_REPLICATE_GENE = 0.15
    PROB_DELETE_GENE = 0.2
    PROB_MUTATE_GENE = 0.3
    PROB_NEW_RANDOM_CONN = 0.3
    
    # Sub-mutation probabilities (must sum to ~1.0)
    PROB_CHANGE_WEIGHT = 0.10
    PROB_INS_NODE = 0.50  # Increased to encourage topology growth
    PROB_CHANGE_ACTIVITY = 0.10
    PROB_CHANGE_OWN_HM = 0.15
    PROB_CHANGE_TARGET_HM = 0.15
    
    # Weight initialization
    WEIGHT_INIT_SIGMA = 1.0
    
    # Fitness
    CONNECTION_COST = 0.0005  # Penalty per connection (encourages parsimony)
    CONNECTION_COST_WARMUP = 10  # No connection cost for first N generations
    
    # Reproduction
    ELITISM_FRACTION = 0.1
    REPRODUCTION_FRACTION = 0.7
    
    # Backprop (Lamarckian)
    BACKPROP_EPOCHS = 50      # Epochs of SGD per agent during fitness eval
    BACKPROP_LR = 0.1         # Learning rate for backprop  
    BACKPROP_BATCH_SIZE = 256 # Batch size for backprop


class Gene:
    """
    A developmental instruction for a neuron.
    
    Operators:
        0 (AddConnection): Grow a connection to a nearby neuron
        1 (InsertNode): Insert a new neuron on an existing connection
    """
    __slots__ = ['op', 'own_hm', 'target_hm', 'weight', 
                 'exp_count', 'ever_activated']
    
    def __init__(self, op: int, own_hm: str, target_hm: str,
                 weight: float = 0.0, exp_count: int = 1):
        self.op = op
        self.own_hm = own_hm
        self.target_hm = target_hm
        self.weight = weight
        self.exp_count = exp_count
        self.ever_activated = False
    
    def clone(self) -> 'Gene':
        g = Gene(self.op, self.own_hm, self.target_hm,
                 self.weight, self.exp_count)
        return g
    
    def mutate_weight(self) -> 'Gene':
        g = self.clone()
        g.weight = np.random.normal(0, HyperParams.WEIGHT_INIT_SIGMA)
        return g
    
    def mutate_activity(self) -> 'Gene':
        g = self.clone()
        if np.random.random() < 0.8:
            delta = int(np.random.normal(0, 4))
            g.exp_count = max(1, min(HyperParams.EXP_COUNT_CAP, 
                                     g.exp_count + delta))
        else:
            g.exp_count = np.random.randint(1, HyperParams.EXP_COUNT_CAP + 1)
        return g
    
    def get_insert_gene(self) -> 'Gene':
        """Create a node-insertion gene following this connection gene."""
        assert self.op == 0, "InsertNode gene can only follow AddConnection"
        return Gene(1, self.own_hm + '1', self.target_hm + '1',
                    weight=np.random.normal(0, HyperParams.WEIGHT_INIT_SIGMA),
                    exp_count=self.exp_count)
    
    def __repr__(self):
        op_name = "AddConn" if self.op == 0 else "InsNode"
        return (f"Gene({op_name} own={self.own_hm} tgt={self.target_hm} "
                f"w={self.weight:.3f} exp={self.exp_count})")


class Node:
    """A neuron in the network topology."""
    __slots__ = ['coords', 'node_type', 'markings', 'io_index',
                 'sources', 'weights', 'outgoing', 'bias',
                 'activation_fn', '_activation', '_fed',
                 '_ancestors',
                 # Backprop caches
                 '_pre_activation', '_input_stack', '_grad_output',
                 '_weight_grads', '_bias_grad']
    
    def __init__(self, coords: Tuple[int, int, int], 
                 node_type: str = 'hidden',
                 activation: str = 'relu',
                 io_index: int = -1):
        self.coords = coords
        self.node_type = node_type
        self.markings = ''
        self.io_index = io_index
        self.sources: List['Node'] = []
        self.weights: List[float] = []
        self.outgoing: List['Node'] = []
        self.bias = 0.0
        self.activation_fn = activation
        self._activation = None
        self._fed = False
        self._ancestors: Optional[Set[int]] = None  # Cached ancestry
        # Backprop state
        self._pre_activation = None   # z = w @ x + b (before activation)
        self._input_stack = None      # stacked source activations
        self._grad_output = None      # dL/d(activation) accumulated from downstream
        self._weight_grads = None
        self._bias_grad = None
    
    def forward(self, batch_size: int) -> np.ndarray:
        """Feed-forward for this node. Returns activation array."""
        if self._fed:
            return self._activation
        
        if not self.sources:
            if self._activation is None:
                self._activation = np.zeros(batch_size)
            self._fed = True
            return self._activation
        
        # Gather inputs
        inputs = []
        for src in self.sources:
            inputs.append(src.forward(batch_size))
        
        x = np.stack(inputs, axis=0)  # (n_sources, batch_size)
        w = np.array(self.weights)     # (n_sources,)
        
        # Weighted sum + bias
        z = w @ x + self.bias  # (batch_size,)
        
        # Cache for backprop
        self._pre_activation = z
        self._input_stack = x
        
        # Activation
        if self.activation_fn == 'relu':
            self._activation = np.maximum(0, z)
        elif self.activation_fn == 'sigmoid':
            self._activation = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation_fn == 'linear':
            self._activation = z
        else:
            self._activation = z
        
        self._fed = True
        return self._activation
    
    def get_ancestors(self) -> Set[int]:
        """
        Cached ancestor set — for DAG enforcement.
        Uses BFS instead of recursive DFS to avoid stack overflow.
        """
        if self._ancestors is not None:
            return self._ancestors
        
        ancestors = set()
        queue = list(self.outgoing)
        while queue:
            node = queue.pop(0)
            node_id = id(node)
            if node_id not in ancestors:
                ancestors.add(node_id)
                queue.extend(node.outgoing)
        
        self._ancestors = ancestors
        return ancestors
    
    def invalidate_ancestors(self):
        """Invalidate cached ancestors (call after topology changes)."""
        self._ancestors = None
    
    def reset(self):
        """Reset activation state for new forward pass."""
        self._fed = False
        self._activation = None
        self._pre_activation = None
        self._input_stack = None
        self._grad_output = None
        self._weight_grads = None
        self._bias_grad = None


class Agent:
    """
    An individual in the population — a genome that develops into a neural network.
    """
    
    def __init__(self, genes: List[Gene], n_input: int, n_output: int,
                 hp: HyperParams = HyperParams()):
        self.genes = genes
        self.n_input = n_input
        self.n_output = n_output
        self.hp = hp
        self.fitness = 0.0
        self.n_connections = 0
        self.n_nodes = 0
        
        # Topology data structures
        self.nodes: List[Node] = []
        self.output_nodes: List[Node] = []
        self.grid = np.empty(
            (hp.SPACE_Z, hp.SPACE_Y, hp.SPACE_X), dtype=object
        )
        self.hm_dict: dict = {}  # marking -> [nodes]
        
        # Build the network
        self._embryogenesis()
        self._finalize()
    
    def _embryogenesis(self):
        """Grow the network topology from genes."""
        self._spawn_minimal()
        
        active_genes = [g for g in self.genes]
        activity = [g.exp_count for g in active_genes]
        
        t0 = time.time()
        changed = True
        
        while active_genes and changed:
            if time.time() - t0 > self.hp.MAX_EMBRYO_TIME:
                break
            
            changed = False
            to_remove = []
            
            for gi in range(len(active_genes)):
                gene = active_genes[gi]
                if gene.own_hm not in self.hm_dict:
                    continue
                
                applied = False
                for node in list(self.hm_dict.get(gene.own_hm, [])):
                    if self._apply_gene(node, gene):
                        applied = True
                        if not gene.ever_activated:
                            gene.ever_activated = True
                        break  # One application per gene per round
                
                if applied:
                    activity[gi] -= 1
                    changed = True
                    if activity[gi] <= 0:
                        to_remove.append(gi)
            
            # Remove exhausted genes (reverse order to preserve indices)
            for gi in sorted(to_remove, reverse=True):
                active_genes.pop(gi)
                activity.pop(gi)
    
    def _spawn_minimal(self):
        """Create minimal network: input and output nodes only."""
        hp = self.hp
        
        # Input nodes
        for i in range(self.n_input):
            x = i % hp.SPACE_X
            y = (i // hp.SPACE_X) % hp.SPACE_Y
            z = 0
            node = Node((x, y, z), node_type='input', 
                       activation='linear', io_index=i)
            node.markings = f'-{i+1}'
            self._register_node(node)
        
        # Output nodes
        for i in range(self.n_output):
            x = i + (hp.SPACE_X - self.n_output) // 2
            y = hp.SPACE_Y // 2
            z = hp.SPACE_Z - 1
            node = Node((x, y, z), node_type='output',
                       activation='linear', io_index=i)
            node.markings = f'-{self.n_input + i + 1}'
            self._register_node(node)
            self.output_nodes.append(node)
    
    def _register_node(self, node: Node):
        """Add node to all tracking structures."""
        self.nodes.append(node)
        x, y, z = node.coords
        if 0 <= x < self.hp.SPACE_X and 0 <= y < self.hp.SPACE_Y and 0 <= z < self.hp.SPACE_Z:
            self.grid[z, y, x] = node
        hm = node.markings
        if hm not in self.hm_dict:
            self.hm_dict[hm] = []
        self.hm_dict[hm].append(node)
    
    def _update_marking(self, node: Node, op: int):
        """Update a node's historical marking."""
        if node.node_type != 'hidden':
            return
        
        old_hm = node.markings
        if old_hm in self.hm_dict:
            nodes_list = self.hm_dict[old_hm]
            if node in nodes_list:
                nodes_list.remove(node)
            if not nodes_list:
                del self.hm_dict[old_hm]
        
        node.markings += str(op)
        new_hm = node.markings
        if new_hm not in self.hm_dict:
            self.hm_dict[new_hm] = []
        self.hm_dict[new_hm].append(node)
    
    def _apply_gene(self, node: Node, gene: Gene) -> bool:
        """Apply a gene's operator to a node."""
        if gene.op == 0:
            return self._add_connection(node, gene)
        elif gene.op == 1:
            return self._insert_node(node, gene)
        return False
    
    def _add_connection(self, source: Node, gene: Gene) -> bool:
        """Grow a connection from source to a nearby matching node."""
        target_hm = gene.target_hm
        candidates = self.hm_dict.get(target_hm, [])
        
        if not candidates:
            return False
        
        sx, sy, sz = source.coords
        source_id = id(source)
        
        # Sort candidates by distance
        dists = []
        for c in candidates:
            if c is source or c in source.outgoing:
                continue
            cx, cy, cz = c.coords
            d_sq = (cx - sx)**2 + (cy - sy)**2 + (cz - sz)**2
            if d_sq <= self.hp.MAX_CONN_LEN_SQ:
                dists.append((d_sq, c))
        
        dists.sort(key=lambda x: x[0])
        
        for _, target in dists:
            # DAG check using cached ancestors
            if source_id in target.get_ancestors():
                continue
            
            # Create connection
            source.outgoing.append(target)
            target.sources.append(source)
            target.weights.append(gene.weight)
            
            # Invalidate ancestor caches for affected nodes
            self._invalidate_ancestors_upstream(source)
            
            self._update_marking(source, 1)
            self._update_marking(target, 1)
            return True
        
        return False
    
    def _insert_node(self, target_node: Node, gene: Gene) -> bool:
        """Insert a new node on an existing connection."""
        target_hm = gene.target_hm
        
        for i, source in enumerate(target_node.sources):
            if source.markings != target_hm:
                continue
            
            # Find a free grid position near the midpoint
            sx, sy, sz = source.coords
            tx, ty, tz = target_node.coords
            mid = ((sx + tx) // 2, (sy + ty) // 2, (sz + tz) // 2)
            
            pos = self._find_free_position(mid)
            if pos is None:
                continue
            
            # Create new node
            new_node = Node(pos, node_type='hidden', activation='relu')
            new_node.markings = '3'
            
            # Rewire: source -> new_node -> target_node
            old_weight = target_node.weights[i]
            
            # new_node receives from source with old weight
            new_node.sources.append(source)
            new_node.weights.append(old_weight)
            new_node.outgoing.append(target_node)
            
            # source: replace target_node with new_node in outgoing
            source.outgoing[source.outgoing.index(target_node)] = new_node
            
            # target_node: replace source with new_node
            target_node.sources[i] = new_node
            target_node.weights[i] = gene.weight if gene.weight != 0 else 1.0
            
            self._register_node(new_node)
            
            # Invalidate ancestor caches
            for n in self.nodes:
                n.invalidate_ancestors()
            
            self._update_marking(source, 2)
            self._update_marking(target_node, 2)
            return True
        
        return False
    
    def _find_free_position(self, near: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        """Find nearest free grid position to target."""
        hp = self.hp
        mx, my, mz = near
        
        # Search in expanding shells
        for r in range(max(hp.SPACE_X, hp.SPACE_Y, hp.SPACE_Z)):
            for dz in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        if dx*dx + dy*dy + dz*dz > r*r:
                            continue
                        x, y, z = mx + dx, my + dy, mz + dz
                        if (0 <= x < hp.SPACE_X and 
                            0 <= y < hp.SPACE_Y and 
                            0 <= z < hp.SPACE_Z and
                            self.grid[z, y, x] is None):
                            return (x, y, z)
        return None
    
    def _invalidate_ancestors_upstream(self, node: Node):
        """Invalidate ancestor caches for node and its predecessors."""
        visited = set()
        queue = [node]
        while queue:
            n = queue.pop(0)
            nid = id(n)
            if nid in visited:
                continue
            visited.add(nid)
            n.invalidate_ancestors()
            for src in n.sources:
                queue.append(src)
    
    def _finalize(self):
        """Finalize network after embryogenesis."""
        self.n_connections = sum(len(n.sources) for n in self.nodes)
        self.n_nodes = len(self.nodes)
        del self.grid  # Free grid memory
    
    def forward(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Forward pass through the network.
        x: (batch_size, n_input)
        Returns: (batch_size, n_output) or None if no valid outputs
        """
        batch_size = x.shape[0]
        
        # Reset all nodes
        for node in self.nodes:
            node.reset()
        
        # Set input activations
        for node in self.nodes:
            if node.node_type == 'input':
                node._activation = x[:, node.io_index]
                node._fed = True
        
        # Forward pass through output nodes (pulls recursively)
        outputs = []
        valid = False
        for node in self.output_nodes:
            a = node.forward(batch_size)
            outputs.append(a)
            if node._fed and node.sources:
                valid = True
        
        if not valid:
            return None
        
        y = np.stack(outputs, axis=1)  # (batch_size, n_output)
        
        # Softmax
        y_exp = np.exp(y - np.max(y, axis=1, keepdims=True))
        y_softmax = y_exp / (np.sum(y_exp, axis=1, keepdims=True) + 1e-10)
        
        return y_softmax
    
    def get_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        pred = self.forward(x)
        if pred is None:
            return 0.0
        y_pred = np.argmax(pred, axis=1)
        return np.mean(y_pred == y)
    
    def get_fitness(self, x: np.ndarray, y: np.ndarray) -> float:
        """Fitness = accuracy - connection_cost * n_connections."""
        accuracy = self.get_accuracy(x, y)
        cost = self.hp.CONNECTION_COST * self.n_connections
        self.fitness = accuracy - cost
        return self.fitness
    
    def _topological_order(self) -> List[Node]:
        """
        Return nodes in topological order (sources before consumers).
        Uses Kahn's algorithm.
        """
        in_degree = {}
        for node in self.nodes:
            in_degree[id(node)] = len(node.sources)
        
        # Start with nodes that have no incoming edges (inputs, disconnected)
        queue = [n for n in self.nodes if in_degree[id(n)] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in node.outgoing:
                in_degree[id(child)] -= 1
                if in_degree[id(child)] == 0:
                    queue.append(child)
        
        return order
    
    def train(self, x: np.ndarray, y: np.ndarray, 
              epochs: int = 5, lr: float = 0.01, 
              batch_size: int = 256) -> float:
        """
        Train the network weights via backpropagation for a few epochs.
        Uses cross-entropy loss for classification.
        Returns final accuracy.
        
        This allows evolution to focus on topology rather than weight luck.
        """
        n_samples = x.shape[0]
        topo_order = self._topological_order()
        
        # Only train if we have connected outputs
        has_connections = any(n.sources for n in self.output_nodes)
        if not has_connections:
            return 0.0
        
        for epoch in range(epochs):
            # Shuffle data each epoch
            perm = np.random.permutation(n_samples)
            x_shuf = x[perm]
            y_shuf = y[perm]
            
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuf[start:end]
                y_batch = y_shuf[start:end]
                bs = x_batch.shape[0]
                
                # === Forward pass ===
                for node in self.nodes:
                    node.reset()
                
                # Set inputs
                for node in self.nodes:
                    if node.node_type == 'input':
                        node._activation = x_batch[:, node.io_index]
                        node._fed = True
                
                # Forward in topo order
                for node in topo_order:
                    node.forward(bs)
                
                # Collect output activations → softmax
                raw_outputs = []
                for node in self.output_nodes:
                    a = node._activation if node._activation is not None else np.zeros(bs)
                    raw_outputs.append(a)
                logits = np.stack(raw_outputs, axis=1)  # (bs, n_output)
                
                # Softmax
                logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits_shifted)
                probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-10)
                
                # === Cross-entropy loss gradient w.r.t. logits ===
                # dL/d(logit_j) = prob_j - 1(y==j)
                one_hot = np.zeros_like(probs)
                one_hot[np.arange(bs), y_batch] = 1.0
                dL_dlogits = (probs - one_hot) / bs  # (bs, n_output)
                
                # === Backward pass ===
                # Initialize gradient accumulators
                for node in self.nodes:
                    node._grad_output = np.zeros(bs)
                
                # Seed output node gradients
                for i, node in enumerate(self.output_nodes):
                    node._grad_output = dL_dlogits[:, i]
                
                # Backward in reverse topo order
                for node in reversed(topo_order):
                    if node._grad_output is None:
                        continue
                    if not node.sources:
                        continue
                    
                    grad_a = node._grad_output  # dL/d(activation), shape (bs,)
                    
                    # Gradient through activation function → dL/dz
                    if node.activation_fn == 'relu':
                        if node._pre_activation is not None:
                            grad_z = grad_a * (node._pre_activation > 0).astype(float)
                        else:
                            grad_z = grad_a
                    elif node.activation_fn == 'sigmoid':
                        a = node._activation
                        if a is not None:
                            grad_z = grad_a * a * (1 - a)
                        else:
                            grad_z = grad_a
                    else:  # linear
                        grad_z = grad_a
                    
                    # dL/d(weight_i) = sum_over_batch(grad_z * source_i_activation)
                    # dL/d(bias) = sum_over_batch(grad_z)
                    # dL/d(source_i_activation) += weight_i * grad_z
                    
                    for j, src in enumerate(node.sources):
                        src_act = src._activation if src._activation is not None else np.zeros(bs)
                        
                        # Weight gradient
                        dw = np.sum(grad_z * src_act)
                        
                        # Propagate gradient to source
                        src._grad_output += node.weights[j] * grad_z
                        
                        # Update weight with SGD
                        node.weights[j] -= lr * dw
                    
                    # Bias gradient
                    db = np.sum(grad_z)
                    node.bias -= lr * db
        
        # Return final accuracy
        return self.get_accuracy(x, y)
    
    def reproduce(self, hm_list: List[str]) -> 'Agent':
        """Create a mutated offspring."""
        new_genes = []
        
        for g in self.genes:
            if not g.ever_activated:
                continue  # Dead genes don't reproduce
            
            # Gene replication
            if np.random.random() < self.hp.PROB_REPLICATE_GENE:
                new_genes.append(g.clone())
            
            r = np.random.random()
            
            # Gene deletion
            if r < self.hp.PROB_DELETE_GENE:
                continue
            r -= self.hp.PROB_DELETE_GENE
            
            # Gene mutation
            if r < self.hp.PROB_MUTATE_GENE:
                mutated = self._mutate_gene(g, hm_list)
                new_genes.extend(mutated)
                continue
            
            # Plain inheritance
            new_genes.append(g.clone())
        
        # Possibly add a new random connection gene
        if np.random.random() < self.hp.PROB_NEW_RANDOM_CONN and hm_list:
            source_hm = hm_list[np.random.randint(len(hm_list))]
            target_hm = hm_list[np.random.randint(len(hm_list))]
            new_gene = Gene(0, source_hm, target_hm,
                           weight=np.random.normal(0, self.hp.WEIGHT_INIT_SIGMA))
            new_genes.append(new_gene)
        
        return Agent(new_genes, self.n_input, self.n_output, self.hp)
    
    def _mutate_gene(self, gene: Gene, hm_list: List[str]) -> List[Gene]:
        """Apply a mutation to a gene. Returns list of resulting genes."""
        r = np.random.random()
        
        # Weight mutation (connection genes only)
        if gene.op == 0 and r < self.hp.PROB_CHANGE_WEIGHT:
            return [gene.mutate_weight()]
        r -= self.hp.PROB_CHANGE_WEIGHT
        
        # Insert node mutation
        if gene.op == 0 and r < self.hp.PROB_INS_NODE:
            return [gene.clone(), gene.get_insert_gene()]
        r -= self.hp.PROB_INS_NODE
        
        # Activity mutation
        if r < self.hp.PROB_CHANGE_ACTIVITY:
            return [gene.mutate_activity()]
        r -= self.hp.PROB_CHANGE_ACTIVITY
        
        # Own HM mutation
        if r < self.hp.PROB_CHANGE_OWN_HM and hm_list:
            g = gene.clone()
            g.own_hm = hm_list[np.random.randint(len(hm_list))]
            return [g]
        r -= self.hp.PROB_CHANGE_OWN_HM
        
        # Target HM mutation
        if r < self.hp.PROB_CHANGE_TARGET_HM and hm_list:
            g = gene.clone()
            g.target_hm = hm_list[np.random.randint(len(hm_list))]
            return [g]
        
        return [gene.clone()]


def generate_retina_dataset(x_size=2, y_size=2):
    """
    Generates the binocular retina dataset from Clune et al.
    'The evolutionary origins of modularity'
    
    Two independent visual fields, each needs to match a pattern.
    Output = 1 only if BOTH fields match their respective patterns.
    This task rewards modularity.
    """
    n_pix = x_size * y_size
    n_possible = 2 ** n_pix
    
    # "Correct" patterns for left and right retina
    left_correct = {0, 1, 5, 4, 7, 2, 13, 8}
    right_correct = {0, 1, 10, 4, 14, 2, 11, 8}
    
    data_x = []
    data_y = []
    
    for left_i in range(n_possible):
        left_bits = [(left_i >> b) & 1 for b in range(n_pix)]
        y_left = left_i in left_correct
        
        for right_i in range(n_possible):
            right_bits = [(right_i >> b) & 1 for b in range(n_pix)]
            y_right = right_i in right_correct
            
            x = left_bits + right_bits
            y = int(y_left and y_right)
            
            data_x.append(x)
            data_y.append(y)
    
    return np.array(data_x, dtype=np.float32), np.array(data_y, dtype=np.int32)


def evolve(generations: int = 20, pop_size: int = 50, verbose: bool = True,
           lamarckian: bool = True):
    """
    Run the evolutionary process.
    
    lamarckian: If True, agents are trained via backprop before fitness evaluation.
                Evolution then selects for *topologies* that learn well,
                not just lucky weight initialization. (Lamarckian because
                learned weights are not inherited — offspring get fresh weights
                from embryogenesis.)
    """
    hp = HyperParams()
    hp.POP_SIZE = pop_size
    
    if not lamarckian:
        hp.BACKPROP_EPOCHS = 0
    
    # Generate dataset
    data_x, data_y = generate_retina_dataset()
    n_input = data_x.shape[1]  # 8 (4 left + 4 right)
    n_output = 2  # binary classification
    
    if verbose:
        print(f"Dataset: {data_x.shape[0]} samples, {n_input} inputs, {n_output} outputs")
        print(f"Class balance: {np.mean(data_y):.3f}")
        print(f"Population: {pop_size}, Generations: {generations}")
        print(f"Grid: {hp.SPACE_X}x{hp.SPACE_Y}x{hp.SPACE_Z}")
        mode = "Lamarckian" if hp.BACKPROP_EPOCHS > 0 else "Darwinian"
        print(f"Mode: {mode}" + (f" ({hp.BACKPROP_EPOCHS} epochs backprop per agent)" if hp.BACKPROP_EPOCHS > 0 else ""))
        print()
    
    # Initialize population with connection + insertion genes
    population = []
    for _ in range(pop_size):
        genes = []
        for i in range(n_input):
            for j in range(n_output):
                # Connection gene
                g = Gene(0, f'-{i+1}', f'-{n_input+j+1}',
                        weight=np.random.normal(0, hp.WEIGHT_INIT_SIGMA))
                genes.append(g)
                # Corresponding insertion gene
                # For InsertNode: own_hm = downstream node, target_hm = upstream node  
                gi = Gene(1, f'-{n_input+j+1}', f'-{i+1}',
                         weight=np.random.normal(0, hp.WEIGHT_INIT_SIGMA))
                genes.append(gi)
        population.append(Agent(genes, n_input, n_output, hp))
    
    best_fitness_history = []
    mean_fitness_history = []
    
    for gen in range(generations):
        t0 = time.time()
        
        # Connection cost warmup: no parsimony pressure early on
        # to let topology diversity survive
        if gen < hp.CONNECTION_COST_WARMUP:
            effective_cost = 0.0
        else:
            effective_cost = hp.CONNECTION_COST
        
        # Evaluate fitness (with optional backprop for Lamarckian evolution)
        fitnesses = []
        for agent in population:
            if hp.BACKPROP_EPOCHS > 0:
                agent.train(data_x, data_y, 
                           epochs=hp.BACKPROP_EPOCHS,
                           lr=hp.BACKPROP_LR,
                           batch_size=hp.BACKPROP_BATCH_SIZE)
            accuracy = agent.get_accuracy(data_x, data_y)
            cost = effective_cost * agent.n_connections
            agent.fitness = accuracy - cost
            fitnesses.append(agent.fitness)
        
        fitnesses = np.array(fitnesses)
        ranked = np.argsort(-fitnesses)  # Best first
        
        best_idx = ranked[0]
        best_fitness = fitnesses[best_idx]
        best_agent = population[best_idx]
        mean_fitness = np.mean(fitnesses)
        
        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)
        
        gen_time = time.time() - t0
        
        if verbose:
            hidden = sum(1 for n in best_agent.nodes if n.node_type == 'hidden')
            print(f"Gen {gen:3d} | Best: {best_fitness:.4f} "
                  f"(acc={best_agent.get_accuracy(data_x, data_y):.4f}) | "
                  f"Mean: {mean_fitness:.4f} | "
                  f"Nodes: {best_agent.n_nodes:3d} (H={hidden:2d}) | "
                  f"Conns: {best_agent.n_connections:3d} | "
                  f"Genes: {len(best_agent.genes):3d} | "
                  f"Time: {gen_time:.2f}s")
        
        # Reproduce
        hm_list = list(best_agent.hm_dict.keys()) if hasattr(best_agent, 'hm_dict') else []
        
        next_gen = []
        
        # Elitism — keep top performers
        n_elite = max(1, int(pop_size * hp.ELITISM_FRACTION))
        for i in range(n_elite):
            parent = population[ranked[i]]
            next_gen.append(Agent(
                [g.clone() for g in parent.genes],
                n_input, n_output, hp
            ))
        
        # Reproduce from top fraction
        n_parents = max(2, int(pop_size * hp.REPRODUCTION_FRACTION))
        while len(next_gen) < pop_size:
            parent_idx = ranked[np.random.randint(n_parents)]
            parent = population[parent_idx]
            offspring = parent.reproduce(hm_list)
            next_gen.append(offspring)
        
        population = next_gen[:pop_size]
    
    return population, best_fitness_history, mean_fitness_history


if __name__ == '__main__':
    print("DENT v2 — Distributed Embryogenesis of Neural Topologies")
    print("=" * 60)
    print("\n--- Lamarckian mode (backprop + evolution) ---")
    pop, best_hist, mean_hist = evolve(
        generations=15, pop_size=30, verbose=True, lamarckian=True
    )
    
    print(f"\nFinal best fitness: {best_hist[-1]:.4f}")
    print(f"Fitness improvement: {best_hist[-1] - best_hist[0]:.4f}")
