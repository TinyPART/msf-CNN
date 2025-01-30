import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce
# Dictionary to track the re-computation frequency of each intermediate tensor element
recomp_MAC = defaultdict(int)
common_MAC = defaultdict(int)

recomp_freq = defaultdict(int)
common_freq = defaultdict(int)

from .analysis.building_blocks import Layer, ConvLayer, DepthwiseConv, PoolingLayer, FusedBlock, Network

# Function to visualize re-computation frequencies as 2D heatmaps for each layer
def visualize_recomp_MAC(layers):
    for layer in layers:
        # Extract the layer's re-computation data from recomp_MAC
        heatmap = layer.output_tensor_compute_freq[:,:,0] * layer.MAC_per_element
        print(f"heatmap size: {heatmap.shape[0]}x{heatmap.shape[1]}")
        
        # Plot the heatmap for this layer
        plt.figure(figsize=(6, 6))
        plt.title(f"Re-computation MAC: {layer.name}")
        plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Re-computation MAC Count')
        plt.xlabel("Width")
        plt.ylabel("Height")
    plt.show()

def visualize_recomp_freq(layers):
    for layer in layers:

        # Create a 2D array to hold re-computation frequencies
        heatmap = layer.output_tensor_compute_freq[:,:,0]
        print(f"heatmap size: {heatmap.shape[0]}x{heatmap.shape[1]}")
        
        
        # Plot the heatmap for this layer
        plt.figure(figsize=(6, 6))
        plt.title(f"Re-computation Frequency: {layer.name}")
        plt.imshow(heatmap, cmap='coolwarm', interpolation='nearest')
        plt.colorbar(label='Re-computation Count')
        plt.xlabel("Width")
        plt.ylabel("Height")
    plt.show()

# Example: Define user-configurable layers
poc_layers = [
    ConvLayer(output_channels=64, kernel_size=3, stride=1),
    PoolingLayer(pool_size=2, stride=2),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1)
]


def MBConv(intput_channel, output_channel, expansion=1, stride=1, padding=1, kernel_size=3):
    intput_channel = int(intput_channel)
    output_channel = int(output_channel)
    return [
        ConvLayer(output_channels=intput_channel * expansion, kernel_size=1, stride=1),
        DepthwiseConv(output_channels=intput_channel * expansion, kernel_size=kernel_size, stride=stride, padding=padding),
        ConvLayer(output_channels=output_channel, kernel_size=1, stride=1),
    ]

w = .35
mobilenetv2_layers = [
    ConvLayer(output_channels=int(32*w), kernel_size=3, stride=2, padding=1),
    *MBConv(32*w, 16*w, 1, 1),
   
    *MBConv(16*w, 24*w, 6, 2),
    *MBConv(24*w, 24*w, 6, 1),

    *MBConv(24*w, 32*w, 6, 2),
    *MBConv(32*w, 32*w, 6, 1),
    *MBConv(32*w, 32*w, 6, 1),

    *MBConv(32*w, 64*w, 6, 2),
    *MBConv(64*w, 64*w, 6, 1),
    *MBConv(64*w, 64*w, 6, 1),
    *MBConv(64*w, 64*w, 6, 1),

    *MBConv(64*w, 96*w, 6, 1),
    *MBConv(96*w, 96*w, 6, 1),
    *MBConv(96*w, 96*w, 6, 1),

    *MBConv(96*w, 160*w, 6, 2),
    *MBConv(160*w, 160*w, 6, 1),
    *MBConv(160*w, 160*w, 6, 1),

    *MBConv(160*w, 320*w, 6, 1),
    ConvLayer(output_channels=int(1280*w), kernel_size=1, stride=1),
    # PoolingLayer(pool_size=7, stride=1),
]

resnet34_layers = [
    ConvLayer(output_channels=64, kernel_size=7, stride=2, padding=3),
    
    #conv2_x
    PoolingLayer(pool_size=3, stride=2, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=64, kernel_size=3, stride=1, padding=1),

    #conv3_x
    ConvLayer(output_channels=128, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=128, kernel_size=3, stride=1, padding=1),

    #conv4_x
    ConvLayer(output_channels=256, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=256, kernel_size=3, stride=1, padding=1),


    #conv5_x
    ConvLayer(output_channels=512, kernel_size=3, stride=2, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),
    ConvLayer(output_channels=512, kernel_size=3, stride=1, padding=1),


]

mcunetv2_vww_5fps = [
    ConvLayer(output_channels=16, kernel_size=3, stride=2, padding=1),
    DepthwiseConv(16,3,1,1),
    ConvLayer(output_channels=8, kernel_size=1, stride=1, padding=0),

    *MBConv(8, 16, 6, 2),
    *MBConv(16, 16, 3, 1),
    *MBConv(16, 16, 3, 1),
    
    *MBConv(16, 24, 3, 2, 3, 7),

    *MBConv(24, 24, 6, 1),
    *MBConv(24, 24, 5, 1, 2, 5),
    *MBConv(24, 40, 6, 2, 3, 7),

    *MBConv(40, 40, 6, 1, 3, 7),

    *MBConv(40, 48, 6, 1, 1, 3),
    *MBConv(48, 48, 4, 1, 1, 3),
    *MBConv(48, 96, 5, 2, 2, 5),

    *MBConv(96, 96, 5, 1, 1, 3),
    *MBConv(96, 96, 4, 1, 1, 3),
    *MBConv(96, 160, 3, 1, 3, 7),
    # PoolingLayer(pool_size=3, stride=1),
]

MCUNet_320KB_ImageNet = [
    ConvLayer(output_channels=16, kernel_size=3, stride=2, padding=1),
    DepthwiseConv(16,3,1,1),
    ConvLayer(output_channels=8, kernel_size=1, stride=1, padding=0),

    *MBConv(8, 16, 3, 2, 3, 7),
    *MBConv(16, 16, 5, 1),
    *MBConv(16, 16, 5, 1, 3, 7),
    
    *MBConv(16, 16, 4, 1, 2, 5),

    *MBConv(16, 24, 5, 2, 2, 5),

    *MBConv(24, 24, 5, 1, 2, 5),

    *MBConv(24, 24, 5, 1, 2, 5),

    *MBConv(24, 40, 5, 2, 1, 3),

    *MBConv(40, 40, 6, 1, 3, 7),
    *MBConv(40, 40, 4, 1, 2, 5),
    *MBConv(40, 48, 5, 1, 2, 5),

    *MBConv(48, 48, 5, 1, 3, 7),
    *MBConv(48, 48, 5, 1, 1, 3),

    *MBConv(48, 96, 6, 2, 1, 3),
    *MBConv(96, 96, 5, 1, 3, 7),
    *MBConv(96, 96, 4, 1, 1, 3),
    *MBConv(96, 160, 5, 1, 3, 7),
    # PoolingLayer(pool_size=3, stride=1),
]

layers = MCUNet_320KB_ImageNet

split_idx_1 = 13
split_idx_2 = 16

block1 = layers[0:split_idx_1]
block2 = layers[split_idx_1:split_idx_2]
block_remain = layers[split_idx_2:]
# Example input tensor (adjust dimensions based on tile size)
input_tensor = np.zeros((176, 176, 3))  # (height, width, channels)

# out_block1 = fused_block1.forward_common(input_tensor)

# print("peak b1 common memory usage:", fused_block1.get_peak_mem())
# print("sum b1 common memory usage:", fused_block1.get_sum_mem())

# # tile_size, stride = calculate_tile_size_and_stride(block1, block_output_size=1)
# # print(f"Calculated tile size: {tile_size}x{tile_size}")
# fused_block1.forward_cache_horizon(input_tensor)

# print("peak b1 memory usage:", fused_block1.memory_usage)
# # print("sum b1 memory usage:", fused_block1.get_sum_mem())

# fused_block2 = FusedBlock(block2, input_tensor)
# out_block2 = fused_block2.forward_common(out_block1)

# print("peak b2 common memory usage:", fused_block2.get_peak_mem())
# print("sum b2 common memory usage:", fused_block2.get_sum_mem())

# Must run first to get original i/o shape
origin_network = Network(layers)
ori_network_mem = origin_network.calc_memory_usage(input_tensor)
print("Original Network memory usage:", ori_network_mem)
print("width multiplier:", w)
# fused_block1 = FusedBlock(block1, input_tensor, 1)
# fused_block2 = FusedBlock(block2, fused_block1.aggregated_output_shape, 1)

# fusion_network = Network([fused_block1, fused_block2, *block_remain])
# fusion_network_mem = fusion_network.calc_memory_usage(input_tensor)
# print("Fusion Network memory usage:", fusion_network_mem)

# def find_minimal_mem_usage_fusion_depth(layers, input_tensor):
#     depth = None
#     mem_usage_lst = []
#     for i,l in enumerate(layers):
#         temp_fusion = FusedBlock(layers[0:i+1], input_tensor)
#         if temp_fusion.tile_size >= input_tensor.shape[0] or temp_fusion.tile_size >= input_tensor.shape[1]:
#             break
#         temp_fusion.forward(input_tensor)
#         mem_usage_lst.append(temp_fusion.memory_usage)
#     if len(mem_usage_lst) != 0:
#         depth = np.argmin(mem_usage_lst) + 1       
#     return depth

# idx = 0
# blocks = []
# block_input_tensor = input_tensor
# fusion_range = []
# while idx < len(layers) - 1:
#     temp_layers = layers[idx:]
#     end_idx = find_minimal_mem_usage_fusion_depth(temp_layers, block_input_tensor)
#     if end_idx is not None:
#         fusion_range.append([idx, idx+end_idx-1])
#         fusion_block = FusedBlock(layers[idx:idx+end_idx], block_input_tensor, block_output_size=1, cache=True)
#         blocks.append(fusion_block)
#         block_input_tensor = np.zeros(fusion_block.aggregated_output_shape)
#     else:
#         end_idx = 1
#         blocks = [*blocks, *layers[idx:idx+end_idx]]
#         block_input_tensor = np.zeros(blocks[-1].common_output_shape)
#     idx += end_idx

# origin_network = Network(layers)
# ori_network_mem = origin_network.calc_memory_usage(input_tensor)
# print("Original Network memory usage:", ori_network_mem)

# from .analysis.memory_first import DPOptimizer
# optimizer = DPOptimizer()
# mem_usage, opt_setting = optimizer.optimize(layers, input_tensor)

## Minimax path solver
# from .analysis.memory_first import MinimaxPathOptimizer
# from .analysis.utils import create_network_from
# optimizer = MinimaxPathOptimizer()
# mem_usage, opt_setting = optimizer.optimize(layers, input_tensor)
# print(f"The minimax path cost from {0} to {len(layers)} is: {mem_usage}")
# print(f"The minimax setting is: {opt_setting}")
# fusion_network = create_network_from(opt_setting, layers, input_tensor)
# fusion_network.reset_compute_counter()
# # _ = [l.set_forward_cache_horizon() for l in blocks]
# fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
# print("Fusion Network memory usage:", fusion_network_mem)
# print("fusion range:", opt_setting)

# Minimaize MAC s.t. Peak MEM Optimizer
from .analysis.memory_first import MinimizeMACstPeakMEMOptimizer
from .analysis.utils import create_network_from
import math
optimizer = MinimizeMACstPeakMEMOptimizer()
PEAK_MEM_TH = 256000
mac_usage, opt_setting = optimizer.optimize(layers, input_tensor, PEAK_MEM_TH)
print(f"The minimal mac s.t. {PEAK_MEM_TH} from {0} to {len(layers)} is: {mac_usage}")
print(f"The optimal setting is: {opt_setting}")
fusion_network = create_network_from(opt_setting, layers, input_tensor)
fusion_network.reset_compute_counter()
# _ = [l.set_forward_cache_horizon() for l in blocks]
fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
print("Fusion Network memory usage:", fusion_network_mem)
print("fusion range:", opt_setting)

# Minimize PEAK MEM s.t. MAC Overhead factor
# from .analysis.memory_first import MinimizePeakMEMstMOFOptimizer
# from .analysis.utils import create_network_from
# import math
# optimizer = MinimizePeakMEMstMOFOptimizer()
# MAC_OVERHEAD_FAC = 1.5
# mem_usage, opt_setting = optimizer.optimize(layers, input_tensor, MAC_OVERHEAD_FAC)
# print(f"The minimal Peak MEM s.t. MAC Overhead {MAC_OVERHEAD_FAC} from {0} to {len(layers)} is: {mem_usage}")
# print(f"The optimal setting is: {opt_setting}")
# fusion_network = create_network_from(opt_setting, layers, input_tensor)
# fusion_network.reset_compute_counter()
# fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
# print("Fusion Network memory usage:", fusion_network_mem)
# print("fusion range:", opt_setting)

fusion_mac = fusion_network.total_mac
common_mac = fusion_network.total_common_mac
redudant_compute = fusion_mac - common_mac
print(f'total:{fusion_mac}, common: {common_mac}, redudant:{redudant_compute}, redudant rate: {redudant_compute / fusion_mac}, overhead factor: {fusion_mac / common_mac}')




# paths = all_paths_below_threshold(graph_adj_matrix, 0, len(layers), 40000)

# for path, max_weight in paths:
#     print(f"Path: {path}, Max Weight: {max_weight}")

# Greedy version
# fusion_network = Network(blocks)
# fusion_network.reset_compute_counter()
# # _ = [l.set_forward_cache_horizon() for l in blocks]
# fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
# print("Fusion Network memory usage:", fusion_network_mem)
# print("fusion range:", fusion_range)


# dummy_block = FusedBlock(layers, input_tensor)
# fusion_network = layers
# layer_in_fusion_block = []
# layers_sorted_by_memory_usage = sorted(layers, key=lambda x: x.memory_usage, reverse=True)

# for l in layers_sorted_by_memory_usage:
#     l_idx = layers.index(l)
#     tem_network = fusion_network


# # Create a fused block with user-defined layers
# fused_block = FusedBlock(layers, input_tensor)

# print("peak common memory usage:", fused_block.get_peak_mem())
# print("sum common memory usage:", fused_block.get_sum_mem())

# # Calculate the tile size for exactly 1 pixel output
# tile_size, stride = calculate_tile_size_and_stride(layers, block_output_size=1)
# print(f"Calculated tile size: {tile_size}x{tile_size}")
# # 
# # breakpoint()



# # Run the fused block forward pass
# fused_block.forward(input_tensor, tile_size, stride)
# # fused_block.forward_cache_horizon(input_tensor, tile_size, stride)

# print("peak memory usage:", fused_block.get_peak_mem())
# print("sum memory usage:", fused_block.get_sum_mem())

# 
# compute_total = 0
# common_compute = 0

# for layer in layers:
#     compute_total += np.sum(layer.output_tensor_compute_freq) * layer.MAC_per_element
#     common_compute += layer.output_tensor_compute_freq.size * layer.MAC_per_element

# common_compute = len(recomp_MAC)
# redudant_compute = compute_total - common_compute

# print(f'total:{compute_total}, common: {common_compute}, redudant:{redudant_compute}, redudant rate: {redudant_compute / compute_total}, overhead factor: {compute_total / common_compute}')

# Visualize re-computation frequencies
# visualize_recomp_freq(layers)