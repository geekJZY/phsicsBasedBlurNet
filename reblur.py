import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
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
from util.metrics import PSNR
from util.util import tensor2im

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
    transformBack = transforms.Normalize(mean=[-0.411,-0.432,-0.45], std=[1,1,1])
    
    model.eval()
    
    flipKernel = Variable(torch.ones(2,1,1,1), requires_grad=False).cuda()
    flipKernel[1] = -flipKernel[1]
    
    # Load Kernel Calculation Module
    KernelModel = flowToKernel()
    # Load blurWithKernel Module
    BlurModel = reblurWithKernel()
    
    avgPSNR = 0.0
    counter = 0
    
    for epoch in range(15):
        for i, sample in enumerate(val_loader):
            counter  = counter + 1
            input_var_1 = torch.autograd.Variable(torch.cat([sample['image0'], sample['image1']],1).cuda(), volatile=True)
            input_var_2 = torch.autograd.Variable(torch.cat([sample['image2'], sample['image1']],1).cuda(), volatile=True)
            # compute output
#             print(input_var_1.data.size())
            output_1 = model(input_var_1)
            output_2 = model(input_var_2)
            # filp the axis direction
            output_1 = F.conv2d(output_1, flipKernel, groups = 2)
            output_2 = F.conv2d(output_2, flipKernel, groups = 2)
            
            output_1 = torch.transpose(output_1, 1, 2)
            output_1 = torch.transpose(output_1, 2, 3)
            output_2 = torch.transpose(output_2, 1, 2)
            output_2 = torch.transpose(output_2, 2, 3)
            ImageKernels = KernelModel.forward((20/16) * output_1, (20/16) * output_2)
 
            blurImg = BlurModel.forward(torch.autograd.Variable(sample['image1']).cuda(), ImageKernels)
            fake_B = (transformBack(blurImg.data[0]).cpu().float().numpy() * 255.0).astype(np.uint8)
            real_B = (transformBack(sample['label'][0]).float().numpy() * 255.0).astype(np.uint8)
            print(fake_B.shape)
            avgPSNR += PSNR(fake_B, real_B)
            print('process image... %s' % str(i))
#             if(counter == 1):
#                 scipy.misc.imsave('blur1.png', np.transpose(transformBack(blurImg.data[0]).cpu().float().numpy(),(1,2,0)))
#                 scipy.misc.imsave('sharp1.png', np.transpose(transformBack(sample['image1'][0]).cpu().float().numpy(),(1,2,0)))
#                 scipy.misc.imsave('blurOrigin.png', np.transpose(transformBack(sample['label'][0]).float().numpy(),(1,2,0)))
#                 return
            if(counter % 20 == 0):
                print('PSNR = %f' % (avgPSNR/counter))
    #         scipy.misc.imsave('outfile{}.png'.format(i), flow2rgb(20 * output.data[0].cpu().numpy(), max_value=25))
    print('PSNR = %f' % (avgPSNR/counter))
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
()