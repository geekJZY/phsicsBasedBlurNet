import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import flow_transforms
from model.FlowNetS import flownets
from model.OpticalFlowToKernel import flowToKernel
from model.reblurWithKernel import reblurWithKernel
from data.optical_flow_dataset import optical_flow_dataset
from data.reblur_dataset import reblurDataSet
import datetime
import numpy as np
import scipy.misc


def main():
    # Load Data and optical flow net
    dataDir = "../flowTestDataset/"

    # Data loading code
#     input_transform = transforms.Compose([
#         flow_transforms.ArrayToTensor(),
#         transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
#         transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
#     ])
    
#     print("=> fetching img pairs in '{}'".format(dataDir))
#     test_set = optical_flow_dataset(
#         dataDir,
#         transform=input_transform
#     )

    test_set = reblurDataSet()
    test_set.initialize("/scratch/user/jiangziyu/test/")
    print('{} samples found'.format(len(test_set)))
    
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
        num_workers=1, pin_memory=True, shuffle=False)
    # create model
    network_data = torch.load('/scratch/user/jiangziyu/flownets_EPE1.951.pth.tar')
    model = flownets(network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    model.eval()
    
    
    # Load Kernel Calculation Module
    KernelModel = flowToKernel()
    # Load blurWithKernel Module
    BlurModel = reblurWithKernel()
    
    for i, sample in enumerate(val_loader):
        input_var_1 = torch.autograd.Variable(torch.cat([sample['image0'], sample['image1']],1).cuda(), volatile=True)
        input_var_2 = torch.autograd.Variable(torch.cat([sample['image2'], sample['image1']],1).cuda(), volatile=True)
        # compute output
        print(input_var_1.data.size())
        output_1 = model(input_var_1)
        output_2 = model(input_var_2)
        output_1 = torch.transpose(output_1, 1, 2)
        output_1 = torch.transpose(output_1, 2, 3)
        output_2 = torch.transpose(output_2, 1, 2)
        output_2 = torch.transpose(output_2, 2, 3)
        print(output_1.data.size())
        ImageKernels = KernelModel.forward((20/16) * output_1, (20/16) * output_2)
        print(ImageKernels.size())
        blurImg = BlurModel.forward(torch.autograd.Variable(sample['image1']).cuda(), ImageKernels)
        print(blurImg.size())
        
#         scipy.misc.imsave('outfile{}.png'.format(i), flow2rgb(20 * output.data[0].cpu().numpy(), max_value=25))

    return

def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
#     print(flow_map)
    flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:,:,2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)

if __name__ == '__main__':
    main()
