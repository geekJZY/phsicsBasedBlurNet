import torch.nn as nn
import torch.nn.functional as F

class reblurWithKernel(nn.Module):
    def __init__(self):
        super(reblurWithKernel, self).__init__()
        self.padding = nn.ReplicationPad2d(1)

    def forward(self, inputImg, Kernels):
        """Input image is in 1*C*H*W form"""
        batchs, channels, height, width = list(inputImg.size())
        output = torch.ones_like(inputImg)
        inputImg = self.padding(inputImg)
        for h in range(height):
            for w in range(width):
                temp = inputImg[0,:,h:h+3,w:w+3]
#                 print(temp.size())
                for channelCnt in range(channels):
                    output[0,channelCnt,h,w] = 0
                    for i in range(3):
                        for j in range(3):
                            output[0,channelCnt,h,w] = output[0,channelCnt,h,w] + temp[channelCnt,i,j]*Kernels[i*3+j,h,w]
        return output
