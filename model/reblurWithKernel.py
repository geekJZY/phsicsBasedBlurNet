import torch.nn as nn
import torch.nn.functional as F
import torch

class reblurWithKernel(nn.Module):
    def __init__(self):
        super(reblurWithKernel, self).__init__()
        self.padding = nn.ReplicationPad2d(16)

    def forward(self, inputImg, Kernels):
        """Input image is in 1*C*H*W form"""
        batchs, channels, height, width = list(inputImg.size())
        output = torch.ones_like(inputImg).cuda()
        inputImg = self.padding(inputImg)
        for h in range(height):
            for w in range(width):
                temp = inputImg[0,:,h:h+33,w:w+33]
#                 print(temp.size())
                for channelCnt in range(channels):
                    output[0,channelCnt,h,w] = 0
                    for i in range(33):
                        for j in range(33):
                            if Kernels[0,i*33+j,h,w] > 1e-4:
                                output[0,channelCnt,h,w] = output[0,channelCnt,h,w] + Kernels[0,i*33+j,h,w]*temp[channelCnt,i,j]
        return output
