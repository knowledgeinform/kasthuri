import torch
import os
import sys
from kasthuri.models.unet import UNet                   #UNet model. From: https://github.com/johschmidt42/PyTorch-2D-3D-UNet-Tutorial

    
#medzoopytorch model library
from lib.medzoo.HighResNet3D import HighResNet3D
from lib.medzoo.Vnet import VNetLight

    
def load_model(network_config, device):
    
    # Specify model
    if network_config["model"] == "UNet":
        print('loading UNet model')
        model = UNet(in_channels=network_config['in_channels'],
                    out_channels=network_config['classes'],
                    n_blocks=network_config['n_blocks'],
                    start_filters=network_config['start_filters'],
                    activation=network_config['activation'],
                    normalization=network_config['normalization'],
                    conv_mode=network_config['conv_mode'],
                    dim=network_config['dim']).to(device)
    
    if network_config["model"] == "smp_UnetPlusPlus":
        print('loading UnetPlusPlus model')
        model = smp.UnetPlusPlus(
                encoder_name=network_config["encoder_name"],         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],   # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                   # model output channels (number of classes in your dataset)
            ).to(device)

    if network_config["model"] == "smp_MAnet":
        print('loading MAnet model')
        model = smp.MAnet(
                encoder_name=network_config["encoder_name"],         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],   # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                   # model output channels (number of classes in your dataset)
            ).to(device)
    
    if network_config["model"] == "smp_PAN":
        print('loading PAN model')
        model = smp.PAN(
                encoder_name=network_config["encoder_name"],         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],   # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                   # model output channels (number of classes in your dataset)
            ).to(device)
    
    if network_config["model"] == "smp_Linknet":
        print('loading Linknet model')
        model = smp.Linknet(
                encoder_name=network_config["encoder_name"],         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],   # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                   # model output channels (number of classes in your dataset)
            ).to(device)
    
    if network_config["model"] == "smp_FPN":
        print('loading FPN model')
        model = smp.FPN(
                encoder_name=network_config["encoder_name"],         # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],   # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                   # model output channels (number of classes in your dataset)
            ).to(device)
    
    if network_config["model"] == "smp_PSPNet":
        print('loading PSPNet model')
        model = smp.PSPNet(
                encoder_name=network_config["encoder_name"],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=network_config["encoder_weights"],     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=network_config["in_channels"],                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=network_config["classes"],                      # model output channels (number of classes in your dataset)
            ).to(device)
    
    if network_config["model"] == 'mzp_VNetLight_3D':
        print('loading VNetLight model')
        model= VNetLight(in_channels=network_config["in_channels"], elu=False, classes=network_config["classes"]).to(device)
    
    if network_config["model"] == "mzp_HighResNet_3D":
        print('loading HighResNet model')
        model= HighResNet3D(in_channels=network_config["in_channels"], classes=network_config["classes"]).to(device)
    
    return model