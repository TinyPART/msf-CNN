import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from functools import reduce

class FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.size = reduce(lambda x,y: x*y, shape)


# HWC format
# Define a base class for layers
class Layer:
    def __init__(self, name, output_channels, kernel_size, stride=1, padding=0, dialation=1):
        self.name = name
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dialation = dialation

        # self.memory_usage = None
        self.MAC_per_element = None
        self.output_tensor_shape = None
        self.output_tensor_compute_freq = None
        self.input_tensor_shape = None
        self.tile_buffer_size = None

        self._common_memory_usage = None
        self._memory_usage = None

        self._total_common_mac = None
        # self._total_mac = None


    def reset_compute_counter(self):
        self.output_tensor_compute_freq[:] = 0

    def forward_common(self, input_tensor):
        output_height = (input_tensor.shape[0]  + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        output_width = (input_tensor.shape[1] + 2 * self.padding - self.dialation* (self.kernel_size - 1) - 1) // self.stride + 1
        self.output_tensor_shape = (output_height, output_width, self.output_channels)
        self.input_tensor_shape = input_tensor.shape
        self.output_tensor_compute_freq = np.ones((output_height, output_width, self.output_channels)) # TODO
        self._common_memory_usage = self.output_tensor_compute_freq.size + input_tensor.size
        self._memory_usage = self._common_memory_usage

        self._total_common_mac = self.MAC_per_element * output_height * output_width * self.output_channels

        return FakeTensor(self.output_tensor_shape)

    
    def get_tile_output_hw(self, input_tensor,i_h, i_w):
        output_height = input_tensor.shape[0] - (self.kernel_size)

        if i_h == 0:
            output_height += self.padding
        if i_h + input_tensor.shape[0] >= self.input_tensor_shape[0]:
            output_height += self.padding
        output_height = output_height // self.stride + 1

        output_width = input_tensor.shape[1] - (self.kernel_size)
        if i_w == 0:
            output_width += self.padding
        if i_w + input_tensor.shape[1] >= self.input_tensor_shape[1]:
            output_width += self.padding
        output_width = output_width // self.stride + 1

        return output_height, output_width
    
    @property
    def memory_usage(self):
        return self._memory_usage

    @property
    def common_memory_usage(self):
        return self._common_memory_usage
       
    @property
    def common_input_size(self):
        return reduce(lambda x,y: x*y, self.input_tensor_shape)
    
    @property
    def common_output_size(self):
        return reduce(lambda x,y: x*y, self.output_tensor_shape)
    
    @property
    def common_output_shape(self):
        return self.output_tensor_shape
    
    @property
    def common_input_shape(self):
        return self.input_tensor_shape
    
    @property
    def buffer_memory_usage(self):
        return self.tile_buffer_size

    @property
    def total_common_mac(self):
        return self._total_common_mac

    def forward(self, input_tensor):
        return self.forward_common(input_tensor)


# Define a convolutional layer
class ConvLayer(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0):
        super().__init__("ConvLayer", output_channels, kernel_size, stride, padding)

    def forward_common(self, input_tensor):
        self.MAC_per_element = input_tensor.shape[2] * (self.kernel_size**2)
        return super().forward_common(input_tensor)
    
    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate convolution by calculating the output dimensions
        # output_height = (input_tensor.shape[0] - (self.kernel_size)) // self.stride + 1
        # output_width = (input_tensor.shape[1] - (self.kernel_size)) // self.stride + 1

        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)
        
        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        self._memory_usage = output_tensor.size + input_tensor.size
        
        # self.tile_buffer_size = output_tensor.size
        self.tile_buffer_size = self.kernel_size * input_tensor.shape[0] * input_tensor.shape[2] # Tile Buffer
        # self.tile_buffer_size = self.kernel_size * self.input_tensor_shape[0] * input_tensor.shape[2] # Full-H-Buffer

        self.MAC_per_element = input_tensor.shape[2] * (self.kernel_size**2)

        # self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1 # TODO
        
        return output_tensor
    
class DepthwiseConv(Layer):
    def __init__(self, output_channels, kernel_size, stride, padding=0, inplace=False):
        super().__init__("DepthwiseConv", output_channels, kernel_size, stride, padding)
        self.inplace = inplace

    def forward_common(self, input_tensor):
        self.MAC_per_element = (self.kernel_size**2)
        o_tensor = super().forward_common(input_tensor)
        if self.inplace:
            self._common_memory_usage = input_tensor.size
            self._memory_usage = input_tensor.size
        return o_tensor
    
    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate convolution by calculating the output dimensions

        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)

        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad

        output_tensor = FakeTensor((output_height, output_width, self.output_channels))

        if self.inplace:
            self._memory_usage = input_tensor.size
        else:
            self._memory_usage = output_tensor.size + input_tensor.size

        # self.tile_buffer_size = output_tensor.size
        self.tile_buffer_size = self.kernel_size * input_tensor.shape[0] * input_tensor.shape[2]
        # self.tile_buffer_size = self.kernel_size * self.input_tensor_shape[0] * input_tensor.shape[2] # Full-H-Buffer
        self.MAC_per_element = (self.kernel_size**2)

        # self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1 # TODO
        
        return output_tensor

# Define a pooling layer
class PoolingLayer(Layer):
    def __init__(self, pool_size, stride, padding=0):
        super().__init__("PoolingLayer", None, pool_size, stride, padding)

    def forward_common(self, input_tensor):
        self.output_channels = input_tensor.shape[2]
        self.MAC_per_element = (self.kernel_size**2)
        return super().forward_common(input_tensor)
    

    def forward(self, input_tensor,i_h=None, i_w=None, acc_pad=None):
        # Simulate pooling by calculating output dimensions
        if i_h is None or i_w is None or acc_pad is None:
            return self.forward_common(input_tensor)

        output_height, output_width = self.get_tile_output_hw(input_tensor, i_h, i_w)

        if i_h != 0:
            i_h += acc_pad
        if i_w != 0:
            i_w += acc_pad
        output_tensor = FakeTensor((output_height, output_width, input_tensor.shape[2]))

        # self.output_tensor_compute_freq[i_h:i_h + output_height, i_w:i_w + output_width, :] += 1 # TODO

        self._memory_usage = output_tensor.size + input_tensor.size
        # self.tile_buffer_size = output_tensor.size
        self.tile_buffer_size = self.kernel_size * input_tensor.shape[0] * input_tensor.shape[2]
        # self.tile_buffer_size = self.kernel_size * self.input_tensor_shape[0] * input_tensor.shape[2] # Full-H-Buffer

        self.MAC_per_element = (self.kernel_size**2)

        return output_tensor

# Define a fused block that can contain arbitrary layers
class FusedBlock:
    def __init__(self, layers, input_tensor, block_output_size=1, cache=False):
        self.layers : list[Layer] = layers
        self.tile_size = None
        self.stride = None
        self.set_block_output_size(block_output_size)
        self.cache = cache
        self.forward = self.forward_no_cache
        self._current_input_tensor = None
        # print("fusion tile size:", self.tile_size)
        # print("fusion stride:", self.stride)
        # self.reset_compute_counter()
        # i = 1
        # for l in self.layers:
        #     l.name = f'{l.name}_{i}'
        #     i += 1
        # self.forward_common(input_tensor)

    def reset_compute_counter(self):
        for l in self.layers:
            l.reset_compute_counter()

    def get_peak_mem(self):
        return reduce(lambda x, y: max(x, y), [l.common_memory_usage for l in self.layers])
    
    def get_sum_mem(self):
        return reduce(lambda x, y: x + y, [l.common_memory_usage for l in self.layers])

    @property
    def memory_usage(self):
        buffer_sum = reduce(lambda x,y: x+y, [l.buffer_memory_usage for l in self.layers])
        # TODO: mem usage with cache is not correct for some case?
        if self.cache:
            # print("cache buffer usage:", buffer_sum)
            return buffer_sum + self.layers[0].common_input_size + self.layers[-1].common_output_size

        else:
            layer_mem_max = max([l.memory_usage for l in self.layers])
            # print("layer peak usage:", layer_mem_max, [l.memory_usage for l in self.layers])
            # print("layer peak usage:", layer_mem_max + self.layers[0].common_input_size + self.layers[-1].common_output_size)
            return layer_mem_max + self.layers[0].common_input_size + self.layers[-1].common_output_size
        

    @property
    def total_common_mac(self):
        return reduce(lambda x,y: x+y, [l.total_common_mac for l in self.layers])

    @property
    def total_fusion_mac(self):
        tile_size = self.tile_size
        tile_stride = self.stride
        
        total_mac = 0

        for l in self.layers:
            input_shape = l.common_input_shape
            # print(f"total_fusion_mac: common_input_shape {input_shape}")
            # print(f"total_fusion_mac: tile size {tile_size}, tile stride {tile_stride}")

            outer_out_h = (input_shape[0] + 2 * l.padding - tile_size) // tile_stride + 1
            outer_out_w = (input_shape[1] + 2 * l.padding - l.kernel_size) // l.stride + 1

            inner_out_h = (tile_size - l.kernel_size) // l.stride + 1
            inner_out_w = 1

            out_ch = l.output_channels

            total_mac += outer_out_h * outer_out_w * inner_out_h * inner_out_w * out_ch * l.MAC_per_element
            # print(f"{outer_out_h} {outer_out_w} {inner_out_h} {inner_out_w} {out_ch} {l.MAC_per_element}")

            tile_size = (tile_size - l.kernel_size) // l.stride + 1
            tile_stride = tile_stride // l.stride

        return total_mac
    
    @property
    def aggregated_output_shape(self):
        return self.layers[-1].common_output_shape
    
    @property
    def aggregated_output_size(self):
        return self.layers[-1].common_output_size

    
    def forward_common(self, input_tensor):
        out_tensor = input_tensor
        for layer in self.layers:
            out_tensor = layer.forward_common(out_tensor)
        return out_tensor

    def forward_no_cache(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block
        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride
        
        # self._current_input_tensor = input_tensor

        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            for j in range(0, width - tile_size + 1, stride):
                tile = input_tensor[i:i+tile_size, j:j+tile_size, :]
                s = 1
                p = 0
                for layer in self.layers:
                    s *= layer.stride
                    p += layer.padding
                    tile = layer.forward(tile, i // s, j // s, p)
                    
        return tile
    
    def set_forward_cache_horizon(self):
        self.forward = self.forward_cache_horizon

    def forward_cache_horizon(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block
        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride

        height, width, _ = input_tensor.shape
        for i in range(0, height - tile_size + 1, stride):
            tile = input_tensor[i:i+tile_size, :, :]
            s = 1
            p = 0
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                p += layer.padding
                tile = layer.forward(tile, i // s, 0, p)
                    
        return tile


    def forward_cache_L_shape(self, input_tensor, tile_size=None, stride=None):
        # Process each tile in the input tensor through the layers in the fused block

        if tile_size is None:
            tile_size = self.tile_size

        if stride is None:
            stride = self.stride

        height, width, _ = input_tensor.shape
        start_h = 0
        start_w = 0
        is_dir_horizon = True
        for i in range(0, height - tile_size + 1, stride):
            tile = input_tensor[i:i+tile_size, i:, :]
            s = 1
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                tile = layer.forward(tile, i // s, i // s)

            tile = input_tensor[i+stride:,i:i+tile_size, :]
            s = 1
            for layer in self.layers:
                # breakpoint()
                s *= layer.stride
                tile = layer.forward(tile, (i  + stride)// s, i // s)
                    
        return tile
    
    def set_block_output_size(self, block_output_size):
        self.tile_size, self.stride = self.calculate_tile_size_and_stride(block_output_size)
        return self.tile_size, self.stride
    
    def calculate_tile_size_and_stride(self, block_output_size):
        tile_size = block_output_size
        stride = block_output_size
        for layer in self.layers[::-1]:
            if isinstance(layer, ConvLayer | PoolingLayer | DepthwiseConv):
                tile_size = (tile_size - 1) * layer.stride + layer.kernel_size
                stride *= layer.stride
        return tile_size, stride
    
class Network:
    def __init__(self, layers) -> None:
        self.layers = layers
        # self.input_tensor = None
    
    def reset_compute_counter(self):
        for l in self.layers:
            l.reset_compute_counter()

    def forward(self, input_tensor):
        output_tensor = input_tensor
        # self.input_tensor = input_tensor
        for l in self.layers:
            output_tensor = l.forward(output_tensor)
            if isinstance(l, FusedBlock):
                output_tensor = np.zeros(l.aggregated_output_shape)
            else:
                output_tensor = np.zeros(output_tensor.shape)
        return output_tensor
    
    def calc_memory_usage(self, input_tensor, ignore_input=True, ignore_output=False):
        self.forward(input_tensor)
        memory_usage = 0
        for i,l in enumerate(self.layers):
            current_mem_usage = l.memory_usage
            if i == 0 and ignore_input:
                current_mem_usage -= input_tensor.size
            elif i == len(self.layers) - 1 and ignore_output:
                current_mem_usage -= l.aggregated_output_size if isinstance(l, FusedBlock) else l.common_output_size
            memory_usage = max(memory_usage, current_mem_usage)          
            
            if isinstance(l, Layer):
                print(f"layer {i} mem usage: {current_mem_usage} \t",
                      f"input shape: {l.common_input_shape} \t size: {l.common_input_size} \t",
                      f"output shape: {l.common_output_shape} \t size: {l.common_output_size}")
            else:
                print(f"layer {i} mem usage: {current_mem_usage} \t",
                      f"output shape: {l.aggregated_output_shape} \t size: {l.aggregated_output_size}")
        return memory_usage

    def get_all_input_shapes(self):
        shapes = []
        for l in self.layers:
            shapes.append(l.common_input_shape)
        return shapes
    
    @property
    def total_mac(self):
        _total_mac = 0
        for i,l in enumerate(self.layers):
            if isinstance(l, Layer):
                _total_mac += l.total_common_mac
            elif isinstance(l, FusedBlock):
                _total_mac += l.total_fusion_mac
            else:
                raise RuntimeError
        return _total_mac
    
    @property
    def total_common_mac(self):
        _total_mac = 0
        for i,l in enumerate(self.layers):
            if isinstance(l, Layer) or isinstance(l, FusedBlock):
                _total_mac += l.total_common_mac
            else:
                raise RuntimeError
        return _total_mac