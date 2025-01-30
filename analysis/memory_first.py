from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer, Network, Layer, FusedBlock
import math
import numpy as np
from .fusion_cost_graph import MemoryUsageEstimator, FusionCostGraphProducer, MACEstimator
from .minimax_memory_optimizer import find_minimax_path, find_shortest_path
# from .find_k_shortest_paths import find_k_shortest_paths_under_weight_sum_threshold
from .utils import from_path_to_fusion_setting

class MinimaxPathOptimizer:
    def __init__(self) -> None:
        pass

    def optimize(self, layers, input_tensor):
        graph_producer = FusionCostGraphProducer(MemoryUsageEstimator)
        fusion_mem_graph = graph_producer.create_graph(layers, input_tensor)
        N = len(layers)
        mem_usage, opt_path = find_minimax_path(fusion_mem_graph, 0, N)
        print(f'Layer Num: {N}, Opt Path: {opt_path}')
        return mem_usage, from_path_to_fusion_setting(opt_path)

# Min(MAC) subject to PeakMEM
class MinimizeMACstPeakMEMOptimizer:
    def __init__(self) -> None:
        pass

    def optimize(self, layers, input_tensor, peak_mem_th=50000):
        graph_producer = FusionCostGraphProducer(MemoryUsageEstimator)
        fusion_mem_graph = np.array(graph_producer.create_graph(layers, input_tensor))
        fusion_mac_graph = np.array(FusionCostGraphProducer(MACEstimator).create_graph(layers, input_tensor))
        above_mem_th_idx = np.nonzero(fusion_mem_graph > peak_mem_th)
        fusion_mac_graph[above_mem_th_idx] = math.inf
        N = len(layers)

        cur_peak_mem = np.max(fusion_mem_graph[fusion_mem_graph != np.inf])
        min_mem = math.inf
        cur_mac = math.inf

        while True:
            fusion_mac, p = find_shortest_path(fusion_mac_graph, 0, N)

            if fusion_mac <= cur_mac and cur_peak_mem < peak_mem_th and p is not []:
                opt_path = p
                min_mem = cur_peak_mem
                cur_mac = fusion_mac

            above_mem_th_idx = np.nonzero(fusion_mem_graph >= cur_peak_mem)
            fusion_mac_graph[above_mem_th_idx] = math.inf
            fusion_mem_graph[above_mem_th_idx] = math.inf   
            temp = fusion_mem_graph[fusion_mem_graph != np.inf]
            if len(temp) == 0:
                break
            cur_peak_mem = np.max(temp)
        
        
        # min_mac, opt_path = find_shortest_path(fusion_mac_graph, 0, N)


        print(f'[MinimizeMACstPeakMEMOptimizer] Layer Num: {N}, Opt Path: {opt_path}, Cost: {cur_mac}')
        # opt_path = [0, 1, 2, 4, 5, 6, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53]
        return cur_mac, from_path_to_fusion_setting(opt_path)
    
# Min Peak MEM subject to MAC Overhead Factor (MOF)
class MinimizePeakMEMstMOFOptimizer:
    def __init__(self):
        pass

    def optimize(self, layers, input_tensor, mof_th):
        graph_producer = FusionCostGraphProducer(MemoryUsageEstimator)
        fusion_mem_graph = np.array(graph_producer.create_graph(layers, input_tensor))
        fusion_mac_graph = np.array(FusionCostGraphProducer(MACEstimator).create_graph(layers, input_tensor))

        common_network = Network(layers)
        common_network.reset_compute_counter()
        common_network.forward(input_tensor)
        common_mac = common_network.total_common_mac
        maximum_mac = common_mac * mof_th
        print(f"common mac: {common_mac}, maximum mac: {maximum_mac}")
        fusion_mac = math.inf
        N = len(layers)
        
        opt_path = []
        
        
        cur_peak_mem = np.max(fusion_mem_graph[fusion_mem_graph != np.inf])
        min_mem = math.inf
        cur_mac = math.inf

        while True:
            fusion_mac, p = find_shortest_path(fusion_mac_graph, 0, N)

            if fusion_mac <= maximum_mac and cur_peak_mem < min_mem and p is not []:
                opt_path = p
                min_mem = cur_peak_mem
                cur_mac = fusion_mac

            above_mem_th_idx = np.nonzero(fusion_mem_graph >= cur_peak_mem)
            fusion_mac_graph[above_mem_th_idx] = math.inf
            fusion_mem_graph[above_mem_th_idx] = math.inf   
            temp = fusion_mem_graph[fusion_mem_graph != np.inf]
            if len(temp) == 0:
                break
            cur_peak_mem = np.max(temp)
             
        # paths = find_k_shortest_paths_under_weight_sum_threshold(fusion_mac_graph, 0, N, maximum_mac)
        return min_mem, from_path_to_fusion_setting(opt_path)


        

class DPOptimizer:
    def __init__(self) -> None:
        self.DP_R = {} # mem usage of sub-networks
        self.DP_p_range = {} # Fused Layer range of R[m, n]
        self.p = {} # mem usage of Fusion block containing layers {L_i,...,L_j}
    
    def _get_fusion_memory_usage(self, m, n):
        if m >= len(self.layers):
            return math.inf
        if m == n:
            return self.layers[m].common_memory_usage
        else:
            block_input_tensor = np.zeros(self.common_input_shapes[m])
            block = FusedBlock(self.layers[m:n+1], block_input_tensor, 1, True)
            if block.tile_size > block_input_tensor.shape[0] or block.tile_size > block_input_tensor.shape[1]:
                return math.inf
            block.forward(block_input_tensor)
            return block.memory_usage

    def get_p(self, m, n):
        if m > n:
            self.p[m, n] = math.inf
        if not (m, n) in self.p:
            self.p[m, n] = self._get_fusion_memory_usage(m, n)
        print(f"get_p: ({m}, {n})", self.p[m , n])
        return self.p[m , n]
        
    def get_dp_r(self, m, n):
        
        if m > n:
            self.DP_R[m, n] = 0
        elif m == n:
            self.DP_R[m, n] = self.get_p(m, n)
        else:
            if not (m, n) in self.DP_R:
                self.DP_R[m, n] = self.R(m, n)
        print("Get DP_R:", self.DP_R[m, n], (m, n))
        return self.DP_R[m, n]

    def R(self, m, n):
        r_min = math.inf
        l_min = None
        for j in range(m, n):
            # print("R:", (m,n,j))
            if j + 1 < n:
                t_r = max(self.get_p(m, j), self.get_dp_r(j + 1, n))
            else:
                t_r = self.get_p(m, j)
            # print("t_r:", t_r, (m, n, j))
            if t_r < r_min:
                r_min = t_r
                l_min = (m, j)
        self.DP_R[m, n] = r_min
        self.DP_p_range[m, n] = l_min
        # print("r_min:", r_min, (m, n))
        return r_min
    
    def optimize(self, layers, input_tensor):
        self.layers = layers
        network = Network(layers)
        network.forward(input_tensor)
        self.common_input_shapes = network.get_all_input_shapes()
        
        N = len(self.layers)
        print("Begin DP Searching...")
        mem_usage = self.R(0, N)
        print("DP Search Finished, min memory usage:", mem_usage)
        opt_setting = []
        layer_mem_usage = []
        
        print("Fetch optimal fusion setting")
        l_min = self.DP_p_range[0, N]
        opt_setting.append(l_min)
        layer_mem_usage.append(self.p[l_min[0], l_min[1]])
        while l_min[1] < N - 1:
            l_min = self.DP_p_range[l_min[1] + 1, N]
            opt_setting.append(l_min)
            layer_mem_usage.append(self.p[l_min[0], l_min[1]])
        print("optimal setting:", opt_setting)
        print("layer_mem_usage:", layer_mem_usage)
        return mem_usage, opt_setting


            


