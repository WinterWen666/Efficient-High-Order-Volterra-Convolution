import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

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


class cha_third_conv(nn.Module):
    def __init__(self, in_cha , out_cha, kernel_size, groups=1):
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
        x = input.reshape(B,C,H*W).reshape(B,C//self.kernel_size, self.kernel_size, H,W)
        x_second = x[:,:,self.PM2_full[:,0]]*x[:,:,self.PM2_full[:,1]]
        x_third = x[:,:,self.PCM_third[:,0]]*x_second[:,:,self.PCM_third[:,1]]
        x = torch.cat([x,x_second,x_third], dim = 2)
        x = x.view(B,self.in_cha,H,W)
        x = self.third_order(x)  
        return x    

class cha_second_conv(nn.Module):
    def __init__(self, in_cha , out_cha, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # self.padding =  padding
        # self.stride = stride
        self.PCMs_r = PM_cross(2, kernel_size)
        self.H_kernel_size = self.PCMs_r[0].shape[0] + 2*kernel_size 
        self.in_cha = in_cha//kernel_size*self.H_kernel_size
        self.out_cha = out_cha
        
        
                 
                                                                   
        self.second_order =  nn.Conv2d(self.in_cha, self.out_cha, 1)
                                   
                                   
                                    
    def forward(self,input):
        B,C,H,W = input.size()
        x = input.view(B,C//self.kernel_size, self.kernel_size, H,W)
        x_cross = x[:,:,self.PCMs_r[0][:,0]]*x[:,:,self.PCMs_r[0][:,1]]
        x = torch.cat([x,x**2,x_cross], dim = 2)
        x = x.view(B,self.in_cha,H,W)
        x = self.second_order(x) 
        
        
        return x
class cha_second_conv_last(nn.Module):
    def __init__(self, in_cha , out_cha, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
       
        self.PCMs_r = PM_cross(2, kernel_size)
        self.H_kernel_size = self.PCMs_r[0].shape[0] + 2*kernel_size 
        self.in_cha = in_cha//kernel_size*self.H_kernel_size
        self.out_cha = out_cha
        
        
                 
                                                                   
        self.second_order =  nn.Linear(self.in_cha, self.out_cha)
                                   
                                   
                                    
    def forward(self,input):
        B,H,W,C = input.size()
        x = input.view(B,H,W,C//self.kernel_size, self.kernel_size)
        x_cross = x[:,:,:,:,self.PCMs_r[0][:,0]]*x[:,:,:,:,self.PCMs_r[0][:,1]]
        x = torch.cat([x,x**2,x_cross], dim = -1)
        x = x.view(B,H,W,self.in_cha)
        x = self.second_order(x) 
        return x                
class H_second_conv(nn.Module):
    def __init__(self, in_cha , H):
        super().__init__()
        self.kernel_size = H
        # self.padding =  padding
        # self.stride = stride
        self.PCMs_r = PM_cross(2, self.kernel_size)
        self.H_kernel_size = self.PCMs_r[0].shape[0] + 2*H
        
        self.weight = nn.Parameter(0.1 * torch.ones(in_cha ,H, self.H_kernel_size,H), 
                                    requires_grad=True) 
        
        
        
                                   
                                   
                                    
    def forward(self,input):
        B,C,H,W = input.size()
        x = input[:,:,:,self.PCMs_r[0][:,0]]*input[:,:,:,self.PCMs_r[0][:,1]]
        x = torch.cat([input,input**2,x],dim = -1)
        x = torch.einsum('bcij,cijo->bcio',x,self.weight)
        
        
        return x