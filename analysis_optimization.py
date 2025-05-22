import numpy as np
from functools import reduce
import logging

from analysis.building_blocks import Layer, ConvLayer, DepthwiseConv, PoolingLayer, FusedBlock, Network

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

def calc_vanilla_and_print_results(layers, input_tensor):
    origin_network = Network(layers)
    ori_network_mem = origin_network.calc_memory_usage(input_tensor)
    print("Original Network memory usage:", ori_network_mem)
    return origin_network
    # print("width multiplier:", w)


# Minimaize MAC s.t. Peak MEM Optimizer
def solve_P2_and_print_results(layers, input_tensor, PEAK_MEM_TH = 64000):
    from analysis.memory_first import MinimizeMACstPeakMEMOptimizer
    from analysis.utils import create_network_from
    import math
    optimizer = MinimizeMACstPeakMEMOptimizer()
    mac_usage, opt_setting = optimizer.optimize(layers, input_tensor, PEAK_MEM_TH)
    print(f"The minimal mac s.t. {PEAK_MEM_TH} from {0} to {len(layers)} is: {mac_usage}")
    print(f"The optimal setting is: {opt_setting}")
    fusion_network = create_network_from(opt_setting, layers, input_tensor)
    fusion_network.reset_compute_counter()
    # _ = [l.set_forward_cache_horizon() for l in blocks]
    fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
    print("Fusion Network memory usage:", fusion_network_mem)
    # print("fusion range:", opt_setting)
    return fusion_network

# Minimize PEAK MEM s.t. MAC Overhead factor
def solve_P1_and_print_results(layers, input_tensor, MAC_OVERHEAD_FAC = 1.5):
    from analysis.memory_first import MinimizePeakMEMstMOFOptimizer
    from analysis.utils import create_network_from
    import math
    optimizer = MinimizePeakMEMstMOFOptimizer()
    mem_usage, opt_setting = optimizer.optimize(layers, input_tensor, MAC_OVERHEAD_FAC)
    print(f"The minimal Peak MEM s.t. MAC Overhead {MAC_OVERHEAD_FAC} from {0} to {len(layers)} is: {mem_usage}")
    print(f"The optimal setting is: {opt_setting}")
    fusion_network = create_network_from(opt_setting, layers, input_tensor)
    fusion_network.reset_compute_counter()
    fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
    print("Fusion Network memory usage:", fusion_network_mem)
    # print("fusion range:", opt_setting)
    return fusion_network

# MCUNet Optimizer
def solve_heuristic_and_print_results(layers, input_tensor):
    from analysis.memory_first import MCUNetOptimizer
    from analysis.utils import create_network_from
    import math
    optimizer = MCUNetOptimizer()
    mem_usage, opt_setting = optimizer.optimize(layers, input_tensor)
    print(f"[MCUNet heuristic Optimizer] The minimal Peak MEM from {0} to {len(layers)} is: {mem_usage}")
    print(f"[MCUNet heuristic Optimizer]  The optimal setting is: {opt_setting}")
    fusion_network = create_network_from(opt_setting, layers, input_tensor)
    fusion_network.reset_compute_counter()
    fusion_network_mem = fusion_network.calc_memory_usage(input_tensor, ignore_output=True)
    print("Fusion Network memory usage:", fusion_network_mem)
    print("fusion range:", opt_setting)
    return fusion_network

def calc_and_print_mac(fusion_network):
    fusion_mac = fusion_network.total_mac
    common_mac = fusion_network.total_common_mac
    redudant_compute = fusion_mac - common_mac
    if (common_mac == 0):
        print("Invalid Network")
        return
    print(f'total:{fusion_mac}, common: {common_mac}, redudant:{redudant_compute}, redudant rate: {redudant_compute / fusion_mac}, overhead factor: {fusion_mac / common_mac}')


model_dict = {"MBV2-w0.35": mobilenetv2_layers, "MN2-vww5": mcunetv2_vww_5fps, "MN2-320K": MCUNet_320KB_ImageNet}
input_tensor_dict = {"MBV2-w0.35": np.zeros((144, 144, 3)), "MN2-vww5": np.zeros((80, 80, 3)), "MN2-320K": np.zeros((176, 176, 3))}

import argparse
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help=" Model: MBV2-w0.35; MN2-vww5; MN2-320K",
                        type=str, default="MBV2-w0.35")
    parser.add_argument("-p", default="P1", type=str, help="Problem: [P1, P2, vanilla, heuristic]")
    parser.add_argument("-c1", default="1.1, 1.2, 1.3, 1.4, 1.5, inf", 
                        type=lambda s: [float(i) for i in s.split(',')], 
                        help="specify the constraint set for the optimization problem P1: F_max. format: n1, n2...n_N")
    parser.add_argument("-c2", default="16, 32, 64, 128, 256", 
                        type=lambda s: [float(i) * 1000 for i in s.split(',')], 
                        help="specify the constraint set for the optimization problem P2: P_max in kB. format: n1, n2...n_N")
    parser.add_argument("--log", default="WARNING", type=str, help="Log Level")

    args = parser.parse_args()

    loglevel = args.log

    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)
    
    model = args.m
    problem = args.p
    c1 = args.c1
    c2 = args.c2

    print(f"Opt. Parameter: Model - {model}, Problem - {problem}, F_max - {c1}, P_max - {c2}")

    model_layers = model_dict[model]
    input_tensor = input_tensor_dict[model]

    def iter_constraints(func, constraints):
        for c in constraints:
            fusion_network = func(model_layers, input_tensor, c)
            calc_and_print_mac(fusion_network)

    if (problem == "P1"):
        iter_constraints(solve_P1_and_print_results, c1)
    elif (problem == "P2"):
        iter_constraints(solve_P2_and_print_results, c2)
    elif (problem == "heuristic"):
        network = solve_heuristic_and_print_results(model_layers, input_tensor)
        calc_and_print_mac(network)
    elif (problem == "vanilla"):
        network = calc_vanilla_and_print_results(model_layers, input_tensor)
        calc_and_print_mac(network)
