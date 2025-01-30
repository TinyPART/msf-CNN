
from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer, Network, Layer, FusedBlock
import math
import numpy as np
import copy

class FusionCostEstimator:
    def __init__(self, network_layers, input_tensor) -> None:
        self.layers = network_layers
        self.input_tensor = input_tensor
        self._cost = {}
        pass

    @property
    def cost(self, m, n):
        return self._cost[m, n]

    @property
    def all_costs(self):
        return copy.copy(self._cost)

class MACEstimator(FusionCostEstimator):
    def __init__(self, network_layers, input_tensor) -> None:
        super().__init__(network_layers, input_tensor)
        self._analysis()

    def _get_fusion_mac(self, m, n):
        funsion_mac = math.inf

        if m >= len(self.layers):
            return math.inf
        if m == n:
            funsion_mac =  self.layers[m].total_common_mac
        else:
            block_input_tensor = np.zeros(self.common_input_shapes[m])
            block = FusedBlock(self.layers[m:n+1], block_input_tensor, 1, True) # DONE calc mem usage of fusion with cache
            if block.tile_size > block_input_tensor.shape[0] or block.tile_size > block_input_tensor.shape[1]:
                return math.inf
            block.forward(block_input_tensor)
            funsion_mac =  block.total_fusion_mac


        return funsion_mac
        
    def _analysis(self):
        network = Network(self.layers)
        network.forward(self.input_tensor)
        self.common_input_shapes = network.get_all_input_shapes()

        for i in range(0, len(self.layers)):
            for j in range(i, len(self.layers)):
                # mem usage of Fusion block containing layers {L_i,...,L_j}
                # print(f"Get Fusion Cost of {i}, {j}")
                self._cost[i, j] = self._get_fusion_mac(i, j)

class MemoryUsageEstimator(FusionCostEstimator):
    def __init__(self, network_layers, input_tensor) -> None:
        super().__init__(network_layers, input_tensor)
        self._analysis()

    def _get_fusion_memory_usage(self, m, n):
        memory_usage = math.inf
        input_tensor_size = 0
        output_tensor_size = 0
        output_shape = None
        if m >= len(self.layers):
            return math.inf
        if m == n:
            memory_usage =  self.layers[m].common_memory_usage
            input_tensor_size = self.layers[m].common_input_size
            output_tensor_size = self.layers[m].common_output_size
        else:
            block_input_tensor = np.zeros(self.common_input_shapes[m])
            input_tensor_size = block_input_tensor.size
            block = FusedBlock(self.layers[m:n+1], block_input_tensor, 1, True) # DONE calc mem usage of fusion with cache
            if block.tile_size > block_input_tensor.shape[0] or block.tile_size > block_input_tensor.shape[1]:
                return math.inf
            block.forward(block_input_tensor)
            memory_usage =  block.memory_usage
            output_tensor_size = block.aggregated_output_size
            output_shape =  block.aggregated_output_shape

        if m == 0:
            memory_usage -= input_tensor_size
            # print(memory_usage, output_shape, self.common_input_shapes[m])
        
        if n == len(self.layers) - 1:
            memory_usage -= output_tensor_size

        return memory_usage
        
    def _analysis(self):
        network = Network(self.layers)
        network.forward(self.input_tensor)
        self.common_input_shapes = network.get_all_input_shapes()

        for i in range(0, len(self.layers)):
            for j in range(i, len(self.layers)):
                # mem usage of Fusion block containing layers {L_i,...,L_j}
                self._cost[i, j] = self._get_fusion_memory_usage(i, j)

class FusionCostGraphProducer:
    def __init__(self, cost_estimator_cls) -> None:

        # self.p_cost = {} # cost / mem usage of Fusion block containing layers {L_i,...,L_j}
        self.estimator_cls = cost_estimator_cls    

    # Graph struct: -1 -> L1 -> L2 -> ... -> L_N
    # Each Node represents layer output (-1 represent network input)
    # Each edge represents the cost (memory consumption, MAC etc) from output_i to output_j
    # For edge (L_i -> L_{i+1}): no fusion
    # Return: adjacency matrix of graph {N+1, N+1}, nodes: (-1, L1...L_N)
    def create_graph(self, network_layers, input_tensor):
        estimator : FusionCostEstimator = self.estimator_cls(network_layers, input_tensor)
        all_costs = estimator.all_costs
        N = len(network_layers)
        adj_matrix = []
        for i in range(0, N):
            row = [math.inf] * (i + 1) + [all_costs[i, j] for j in range(i, N)]
            adj_matrix.append(row)
        adj_matrix.append([math.inf] * (N + 1))
        return adj_matrix




