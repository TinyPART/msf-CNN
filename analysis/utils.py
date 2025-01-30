from .building_blocks import ConvLayer, DepthwiseConv, PoolingLayer, Network, FusedBlock
import numpy as np

# Function to calculate the minimum tile size required to output exactly N pixel
def calculate_tile_size_and_stride(layers, block_output_size=1):
    tile_size = block_output_size
    stride = block_output_size
    for layer in layers[::-1]:
        if isinstance(layer, ConvLayer | PoolingLayer | DepthwiseConv):
            tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
            stride *= layer.stride
    return tile_size, stride


def create_network_from(fusion_setting, layers, input_tensor):
    block_input_tensor = input_tensor
    blocks = []
    for s in fusion_setting:
        if s[0] == s[1]:
            blocks.append(layers[s[0]])
            block_input_tensor = np.zeros(layers[s[0]].common_output_shape)
        else:
            fusion_block = FusedBlock(layers[s[0]:s[1]+1], block_input_tensor, block_output_size=1, cache=True)
            blocks.append(fusion_block)
            block_input_tensor = np.zeros(fusion_block.aggregated_output_shape)
    return Network(blocks)

def from_path_to_fusion_setting(path):
    setting = []
    for i in range(0, len(path) - 1):
        setting.append((path[i], path[i+1] - 1))
    return setting