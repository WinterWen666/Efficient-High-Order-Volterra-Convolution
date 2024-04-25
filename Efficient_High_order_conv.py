import numpy as np
import torch
from torch.autograd import Function
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def PM_creation( N, input_size, TRM):
    index = []
    PM = torch.arange(input_size)[:,None]
    TPM = []
    TPM.append(PM)
    PM_full = [] 
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
        TPM.append(PM)
    for k in range(len(TRM)):
        PM = TPM[len(TRM[k])-1]
        PM_full.append(np.repeat(PM, TRM[k], axis =1))   
    PM_full = torch.vstack(PM_full)     
    return PM_full


def PCM_creation(PM_full_r_1, PM_full_r):
    PCM_r = []
    PCMs_r = []
    for i in range(PM_full_r.shape[1]):
        PM_deleted = torch.cat((PM_full_r[:, :i], PM_full_r[:, i + 1:]), dim=1)
        for row_r in PM_deleted:
            Position = torch.where(torch.all(PM_full_r_1 ==row_r, axis = 1))
            PCM_r.append(torch.tensor(Position))
        PCM_r = torch.vstack(PCM_r)
        PCM_r = torch.concat((PM_full_r[:,i][:,None], PCM_r), axis = 1)
        PCMs_r.append(PCM_r) 
        PCM_r = []
    return  PCMs_r  


class HT_creation(Function):
        @staticmethod
        def forward(ctx, input, kernel_size, stride, padding, 
                  TPCMs_r):
            input = input.to(device)
            shape = input.shape
            grad_move = []
            grad_move.append(0)
            x_r_1_move =[]
            x_r_1_move.append(0)
            HT = []
            unfold = nn.Unfold(kernel_size = kernel_size,  padding = padding, stride = stride)
            Col = unfold(input)
            weight_shape = []
            
            ori_col = int(kernel_size[0]*kernel_size[1])
            Col = Col.view(shape[0], shape[1], ori_col, Col.shape[-1])
            
            for i in range(len(TPCMs_r)):  
                if i == 0:
                    grad_move.append(grad_move[i]+TPCMs_r[i][0].shape[0])
                    weight_shape.append((ori_col, ori_col))
                    HT.append(Col[:,:,TPCMs_r[i][0][:,0],:]*Col[:,:,TPCMs_r[i][0][:,1],:])
                    x_r_1_move.append(x_r_1_move[i]+Col.shape[2])
                else:
                    grad_move.append(grad_move[i]+TPCMs_r[i][0].shape[0])
                    weight_shape.append((TPCMs_r[i-1][0].shape[0], ori_col))
                    HT.append(Col[:,:,TPCMs_r[i][0][:,0],:]*HT[i-1][:,:,TPCMs_r[i][0][:,1],:])
                    x_r_1_move.append(x_r_1_move[i]+TPCMs_r[i-1][0].shape[0])
            HT = torch.cat(HT, dim = 2)     
            if len(TPCMs_r) == 1:
                # ctx.back_terms = Col
                ctx.save_for_backward(Col)
            else:
                # ctx.back_terms = torch.cat((Col,HT[:,:,0:grad_move[-2],:]), dim=2 )
                ctx.save_for_backward(torch.cat((Col,HT[:,:,0:grad_move[-2],:]), dim=2 ))
            ctx.grad_move = grad_move
            ctx.x_r_1_move = x_r_1_move
            ctx.TPCMs_r = TPCMs_r
            
            ctx.weight_shape = weight_shape
            ctx.kernel_size =kernel_size
            ctx.stride = stride
            ctx.padding = padding
            ctx.shape =shape
            
            return HT
        @staticmethod
        def backward(ctx, grad):
            (back_terms,) = ctx.saved_tensors
            
            # back_terms = ctx.back_terms
            weight_shape = ctx.weight_shape
            TPCMs_r = ctx.TPCMs_r
            kernel_size = ctx.kernel_size
            stride = ctx.stride
            padding = ctx.padding
            shape = ctx.shape
            grad_move = ctx.grad_move
            x_r_1_move = ctx.x_r_1_move
            grad_input  = None
            
            for i in range(len(TPCMs_r)):
                Al_grad = torch.zeros((grad.shape[0], grad.shape[1],
                                          weight_shape[i][0], weight_shape[i][1],grad.shape[-1])).to(device)
                if i == 0: 
                    Al_grad[:,:, TPCMs_r[0][0][:,0],TPCMs_r[0][0][:,1],:] = grad[:,:,grad_move[0]:grad_move[1],:]
                    # Al_grad[:,:, PCMs_r[0][:,1],PCMs_r[0][:,0],:] += grad[:,:,grad_move[0]:grad_move[1],:]
                    Al_grad = Al_grad + Al_grad.transpose(2,3)
                    grad_input = torch.einsum('bcikj,bckj->bcij',Al_grad, 
                                              back_terms[:,:,x_r_1_move[0]:x_r_1_move[1],:])
                else:
                    Al_grad[:,:, TPCMs_r[i][0][:,1], TPCMs_r[i][0][:,0],:] = grad[:,:,grad_move[i]:grad_move[i+1],:]
                    for j in range(i+1):
                        Al_grad[:,:, TPCMs_r[i][j+1][:,1], TPCMs_r[i][j+1][:,0],:] +=grad[:,:,grad_move[i]:grad_move[i+1],:]
                    grad_input += torch.einsum('bckij,bckj->bcij',Al_grad, 
                                                back_terms[:,:,x_r_1_move[i]:x_r_1_move[i+1],:])
            grad_input = grad_input.reshape(grad_input.shape[0], 
                                          grad_input.shape[1]*grad_input.shape[2], grad_input.shape[-1])    
            fold = nn.Fold((shape[-2], shape[-1]), kernel_size =kernel_size, stride = stride, padding = padding )
            grad_input = fold(grad_input)
                       
            return grad_input, None, None, None, None, None

class high_order_input(nn.Module):
    def __init__(self, kernel_size, stride, padding, TPCMs_r):
        super().__init__()
        self.func = HT_creation
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.TPCMs_r = TPCMs_r
       
       
    def forward(self, input): 
        return self.func.apply(input, 
                          self.kernel_size, 
                          self.stride,
                          self.padding,
                          self.TPCMs_r
                          )


class HO_conv(nn.Module):
    def __init__(self, in_chan , out_chan, kernel_size, stride, padding, groups, TPCMs_r):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding =  padding
        self.stride = stride
        self.TPCMs_r = TPCMs_r
        self.H_kernel_size = 0
        for i in range(len(self.TPCMs_r)):
            
            size = self.TPCMs_r[i][0].shape[0]
           
            self.H_kernel_size += size
        self.conv =  nn.Sequential( high_order_input( self.kernel_size, 
                                                self.stride,
                                                self.padding,
                                                self.TPCMs_r),
                                    nn.Conv2d(in_chan, out_chan, 
                                              kernel_size=(self.H_kernel_size,1), 
                                              stride = 1,
                                              groups = groups, bias = False))
    def forward(self,input):
        in_shape = input.shape
        x = self.conv(input)
        x =  x.reshape(x.shape[0],
                        x.shape[1], 
                      (in_shape[-2]-self.kernel_size[0]+ 2* self.padding[0])//self.stride[0] + 1, 
                      (in_shape[-1]-self.kernel_size[1]+ 2* self.padding[1])//self.stride[1] + 1 )
        return x