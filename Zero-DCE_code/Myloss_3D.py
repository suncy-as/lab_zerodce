import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
torch.set_default_dtype(torch.float64)


class L_color(nn.Module):
#    
#    感觉应该像是对三个通道的东西做平均，让三个通道差不多吧。感觉用不到
#    事实上最后也没有用这个loss

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k

    
    
class L_spa(nn.Module):
    """
    这个应该是空间一致性损失，要大改，从上下左右变成上下左右前后6块区块，然后求loss
    """

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # 确定需要计算的区域，加入cuda中
        kernel_left = torch.DoubleTensor( [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[-1,1,0],[0,0,0]],
                                          [[0,0,0],[0,0,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.DoubleTensor( [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,-1],[0,0,0]],
                                          [[0,0,0],[0,0,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)
        
        kernel_up = torch.DoubleTensor( [[[0,0,0],[0,-1,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],
                                          [[0,0,0],[0,0,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.DoubleTensor( [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,0,0]],
                                          [[0,0,0],[0,-1,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)
        
        kernel_front = torch.DoubleTensor( [[[0,0,0],[0,0,0],[0,0,0]],[[0,-1,0],[0,1,0],[0,0,0]],
                                          [[0,0,0],[0,0,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_back = torch.DoubleTensor( [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,1,0],[0,-1,0]],
                                          [[0,0,0],[0,0,0],[0,0,0]]]).cuda().unsqueeze(0).unsqueeze(0)

#         kernel_left = torch.FloatTensor( [[[0,0,0],[0,0,0],[0,0,0]],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)

        
        # 这里四个数组是自己设计的卷积核，具体形式见上面的kernel_left等
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_front = nn.Parameter(data=kernel_front, requires_grad=False)
        self.weight_back = nn.Parameter(data=kernel_back, requires_grad=False)
        self.pool = nn.AvgPool3d(4)
        
    def forward(self, org , enhance ):
        # L_spa的输入参数：enhance.shape:torch.Size([1, 3, 32, 32, 32]), lowlight.shape:torch.Size([1, 1, 32, 32, 32])
        
        b,c,h,w,l = org.shape
        
    
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)    

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        
#         print('enhance_pool.shape:{}, org_pool.shape:{}'.format(enhance_pool.shape, org_pool.shape))
#         weight_diff = torch.max(torch.FloatTensor([1]).cuda() + 
#             10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),
#                                 torch.FloatTensor([0.5]).cuda())
#         # max(1+ 10000* min((org_pool-0.3),0), 0.5)    纯纯经验公式
        
#         E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)
        # torch.sign是一个符号函数，使得正数变成1，负数变成-1
        # torch.sign（enhance_pool - 0.5） * (enhance_pool-org_pool),这个E1最后好像没用了
        # torch.sign（enhance_pool - 0.5）是使得enhance_pool的亮部和暗部成为一个加权系数
        
        # 这里写成卷积的形式应该是为了便于并行计算
        D_org_left = F.conv3d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv3d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv3d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv3d(org_pool , self.weight_down, padding=1)
        D_org_front = F.conv3d(org_pool , self.weight_front, padding=1)
        D_org_back = F.conv3d(org_pool , self.weight_back, padding=1)

        D_enhance_left = F.conv3d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv3d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv3d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv3d(enhance_pool , self.weight_down, padding=1)
        D_enhance_front = F.conv3d(enhance_pool , self.weight_front, padding=1)
        D_enhance_back = F.conv3d(enhance_pool , self.weight_back, padding=1)
        
        # 求二范数
        D_left = torch.pow(D_org_left - D_enhance_left,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        D_front = torch.pow(D_org_front - D_enhance_front,2)
        D_back = torch.pow(D_org_back - D_enhance_back,2)
        E = (D_left + D_right + D_up +D_down + D_front + D_back)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
        
    
    
    
    
class L_exp(nn.Module):
    '''
    曝光控制损失
    '''
    def __init__(self,patch_size,mean_val):        # 16   0.6
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool3d(patch_size)
        self.mean_val = mean_val  # 0.6
    def forward(self, x ):
        #  A:torch.Size([1, 24, 32, 32, 32]) 新加了一个维度的diff，符号为l

        b,c,h,w,l = x.shape    #batch, channel, height, width, length
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.DoubleTensor([self.mean_val] ).cuda(),2))
        return d
        
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        #  A:torch.Size([1, 24, 32, 32, 32]) 新加了一个维度的diff，符号为l
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        l_x = x.size()[4]
        
        count_h =  (x.size()[2]-1) * x.size()[3] * x.size()[4]
        count_w = x.size()[2] * (x.size()[3] - 1) * x.size()[4]
        count_l = x.size()[2] * x.size()[3] * (x.size()[4] - 1)
        
        h_tv = torch.pow( ( x[:,:,1:,:,:] - x[:,:,:h_x-1,:,:] ),2).sum()    #求diff
        w_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:w_x-1,:]),2).sum()
        l_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:l_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w+l_tv/count_l)/batch_size
    
    
    
    
    
    
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3
