import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

# PM_cross is the combination of position for non-identical high order terms.
# PM_Full_creation is the all combination of position for high order terms.
# PCM_creation is used to order over 2. One of the PCMs can be used to acheieve high order convolution. 
# The others of the PCMs are used for customizing backpropagation.  

def PM_cross( N, input_size):
    index = []
    PM = torch.arange(input_size)[:,None]
    
    PM_cross = [] 
    for i in range(N-1):
        Nc = int(PM.shape[1])
        for j in range(input_size-Nc):
            j = j+1
            p_N1 = (PM[:,Nc-1] + j)[:, None]
            index_chosen = torch.argwhere(p_N1 < input_size)
            index_final = torch.concat((PM[index_chosen[:,0]],p_N1[index_chosen[:,0]]), axis = 1)
            index.append(index_final)
            p_N1 = p_N1 - j
        PM = torch.vstack(index)
        index = []
        PM_cross.append(PM)
    
    return PM_cross
    
def PM_Full_creation( r, L, TRM_r):
    PM_full = []
    NPM = []
    TNPM_r = []
    NPM_r = torch.arange(L)[:,None]
    TNPM_r.append(NPM_r)
    for i in range(r-1):      
        Nc = int(NPM_r.shape[1])
        for j in range(L-Nc):
            j = j+1
            NP_r = (NPM_r[:,Nc-1] + j)[:, None]
            Chosen = torch.argwhere(NP_r < L)
            PM_ind = torch.concat((NPM_r[Chosen[:,0]],NP_r[Chosen[:,0]]), axis = 1)
            NPM.append(PM_ind)
            NP_r = NP_r - j   
        NPM_r = torch.vstack(NPM)
        NPM = []
        TNPM_r.append(NPM_r)
    for k in range(len(TRM_r)):
        PM = TNPM_r[len(TRM_r[k])-1]
        PM_full.append(np.repeat(PM, TRM_r[k], axis =1))   
    PM_full = torch.vstack(PM_full)      
    return  PM_full
    
def PCM_creation(PM_full_r_1, PM_full_r):
    PCM_r = []
    PCMs_r = []
    for i in range(PM_full_r.shape[1]):
        PM_deleted = torch.concat((PM_full_r[:, :i], PM_full_r[:, i + 1:]),axis = 1)
        for row_r in PM_deleted:
            Position = torch.where(torch.all(PM_full_r_1 ==row_r, axis = 1))
            PCM_r.append(Position[0])
        PCM_r = torch.vstack(PCM_r)
        PCM_r = torch.concat((PM_full_r[:,i][:,None], PCM_r), axis = 1)
        PCMs_r.append(PCM_r) 
        PCM_r = []
    return PCMs_r 

    
# Second_GCHOC and Third_GCHOC are second order GCHOC and Third order GCHOC, respectively. Second_order_conv2d is 2d second order convolution.    
# Concatenate can be replaced by for loop(more conv2d layers need to be defined) if memory is not enough. However, it will slow down the training.     

class Second_GCHOC(nn.Module):
    def __init__(self, in_cha , out_cha, cha_group, kernel_size=1,stride=1,groups=1):
        super().__init__()
        self.kernel_size = cha_group
       
       
        self.PCMs_r = PM_cross(2, cha_group)
        self.H_kernel_size = self.PCMs_r[0].shape[0] + 2*cha_group 
        self.in_cha = in_cha//cha_group*self.H_kernel_size
        
        self.padding = kernel_size//2
        if kernel_size ==1:
            self.second_order =  nn.Conv2d(self.in_cha, out_cha,1)
        else:
            self.second_order =  nn.Conv2d(self.in_cha, out_cha,
                                           kernel_size = kernel_size, stride = stride,
                                           padding=self.padding, groups = groups)
                         
                                    
    def forward(self,input):
        B,C,H,W = input.size()
        x = input.view(B,self.kernel_size,C//self.kernel_size, H,W)
        x_cross = x[:,self.PCMs_r[0][:,0]]*x[:,self.PCMs_r[0][:,1]]
        x = torch.cat([x,x**2,x_cross], dim = 1)
        x = x.view(B,self.in_cha,H,W)
        x = self.second_order(x) 
        return x
class Third_GCHOC(nn.Module):
    def __init__(self, in_cha , out_cha, kernel_size,stride=1,groups=1):
        super().__init__()
        self.kernel_size = kernel_size
        TRM2 = [[2],[1,1]]  # Second order TRM
        TRM3 = [[3],[1,2],[2, 1], [ 1, 1, 1]]  # Third order TRM 
        # self.padding =  padding
        # self.stride = stride
        self.PM2_full = PM_Full_creation(2, kernel_size, TRM2)
        PM3_full = PM_Full_creation(3, kernel_size, TRM3)
        self.PCM_third = PCM_creation(self.PM2_full,PM3_full)[0]
        self.H_kernel_size = kernel_size+self.PM2_full.shape[0] + PM3_full.shape[0]
        self.in_cha = in_cha//kernel_size*self.H_kernel_size
        self.out_cha = out_cha                                                                   
        self.third_order =  nn.Conv2d(self.in_cha, self.out_cha, 1, groups = groups)                                
    def forward(self,input):
        B,C,H,W = input.size()
        x = input.view(B,self.kernel_size,C//self.kernel_size, H,W)
        x_second = x[:,self.PM2_full[:,0]]*x[:,self.PM2_full[:,1]]
        x_third = x[:,self.PCM_third[:,0]]*x_second[:,self.PCM_third[:,1]]
        x = torch.cat([x,x_second,x_third], dim = 1)
        x = x.view(B,self.in_cha,H,W)
        x = self.third_order(x)  
        return x 
        
        
class Second_order_conv2d(nn.Module):
    def __init__(self,in_cha, out_cha, kernel_size,stride, padding,  dilation=1, groups=1):
        super().__init__()
        self.k =  kernel_size
        self.s = stride
        self.p = padding
        self.d = dilation
        self.out_cha = out_cha
        self.unfold =nn.Unfold(kernel_size =kernel_size , dilation=dilation, padding=padding, stride=stride)
        self.vol_size = kernel_size**2
        self.PCMs_r = PM_cross(2, self.vol_size)
        self.kernel_size = self.PCMs_r[0].shape[0] + 2*kernel_size**2
        self.second_conv = nn.Conv2d(in_cha, self.out_cha, (self.kernel_size,1),bias = False)                             
    def forward(self,input):
        B,ori_C,H,W = input.size()
        out_H = (H + 2*self.p - self.d*(self.k-1)-1)//self.s + 1
        out_W = (W + 2*self.p - self.d*(self.k-1)-1)//self.s + 1
        if self.k == self.s:
            x = input.view(B,ori_C,self.s,H//self.s,self.s,W//self.s)
            x = x.transpose(3,4).view(B,ori_C,self.s**2,H//self.s*W//self.s)
        else:    
            x = self.unfold(input)
            B,C,L = x.size()
            x = x.view(B,ori_C,self.vol_size , L)
        x_cross = x[:,:,self.PCMs_r[0][:,0]]*x[:,:,self.PCMs_r[0][:,1]]
        x = torch.cat([x,x**2,x_cross], dim = 2)
        x = self.second_conv(x) 
        x = x.view(B,self.out_cha, out_H, out_W)
        return x                
