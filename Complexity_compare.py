

import numpy as np
from timeit import Timer 
from memory_profiler import profile
from scipy.io import savemat
                                            
                               

TRM2 = [[2],[1,1]]  # Second order TRM
TRM3 = [[3],[1,2],[2, 1], [ 1, 1, 1]]  # Third order TRM 
TRM4 = [[4],[1,3],[3, 1],[2,2] ,[2, 1, 1], [1, 2, 1], [1, 1, 2], [1, 1, 1, 1]] # Fourth order TRM 

def PM_Full_creation( r, L, TRM_r):
    PM_full = []
    NPM = []
    TNPM_r = []
    NPM_r = np.arange(L)[:,None]
    TNPM_r.append(NPM_r)
    for i in range(r-1):      
        Nc = int(NPM_r.shape[1])
        for j in range(L-Nc):
            j = j+1
            NP_r = (NPM_r[:,Nc-1] + j)[:, None]
            Chosen = np.argwhere(NP_r < L)
            PM_ind = np.concatenate((NPM_r[Chosen[:,0]],NP_r[Chosen[:,0]]), axis = 1)
            NPM.append(PM_ind)
            NP_r = NP_r - j   
        NPM_r = np.vstack(NPM)
        NPM = []
        TNPM_r.append(NPM_r)
    for k in range(len(TRM_r)):
        PM = TNPM_r[len(TRM_r[k])-1]
        PM_full.append(np.repeat(PM, TRM_r[k], axis =1))   
    PM_full = np.vstack(PM_full)      
    return  PM_full




def PCM_creation(PM_full_r_1, PM_full_r):
    PCM_r = []
    PCMs_r = []
    for i in range(PM_full_r.shape[1]):
        PM_deleted = np.concatenate((PM_full_r[:, :i], PM_full_r[:, i + 1:]),axis = 1)
        for row_r in PM_deleted:
            Position = np.where(np.all(PM_full_r_1 ==row_r, axis = 1))
            PCM_r.append(Position)
        PCM_r = np.vstack(PCM_r)
        PCM_r = np.concatenate((PM_full_r[:,i][:,None], PCM_r), axis = 1)
        PCMs_r.append(PCM_r) 
        PCM_r = []
    return PCMs_r  
 


# del PM2_full, PM3_full, #PM4_full
@profile
def EVC_forward(input, B, C, L, out_cha, TPCMs_r):
    TW = []
    THR = []
    back_shapes = []
    THR.append(input)
    for i in range(len(TPCMs_r)):
        if i == 0:
            back_shapes.append((out_cha, C, L, L))
            TW.append(np.ones((out_cha, C, TPCMs_r[i][0].shape[0])))
            THR.append(THR[0][:,:,TPCMs_r[i][0][:,0]]*THR[i][:,:,TPCMs_r[i][0][:,1]])
            out = np.einsum('BCL,OCL -> BO',THR[1], TW[0])
            
        else:
            back_shapes.append((out_cha, C, L, TPCMs_r[i-1][0].shape[0]))
            TW.append(np.ones((out_cha, C, TPCMs_r[i][0].shape[0])))
            THR.append(THR[0][:,:,TPCMs_r[i][0][:,0]]*THR[i][:,:,TPCMs_r[i][0][:,1]])
            out += np.einsum('BCL,OCL -> BO',THR[i+1], TW[i])       
    return out,  TW, THR, back_shapes



@profile
def EVC_backward(gradient, TW, THR, back_shapes, TPCMs_r):
    
    for i in range(len(TPCMs_r)):
        if i == 0:
            W_back = np.zeros(back_shapes[0])
            W_back[:,:,TPCMs_r[i][0][:,0],TPCMs_r[i][0][:,1]] = TW[i]
            # W_back[:,:,TPCMs_r[i][0][:,1],TPCMs_r[i][0][:,0]] = W_back[:,:,TPCMs_r[i][0][:,1],TPCMs_r[i][0][:,0]] + TW[i]
            W_back +=W_back.transpose(0,1,3,2)
            grad = np.einsum('BO,OCIL,BCL -> BCI',gradient, W_back, THR[0])
        else:
            W_back = np.zeros(back_shapes[i])
            for j in range(len(TPCMs_r[i])):
                W_back[:,:,TPCMs_r[i][j][:,0],TPCMs_r[i][j][:,1]] += TW[i]
            grad += np.einsum('BO,OCIL,BCL -> BCI',gradient, W_back, THR[i])    
    return grad





@profile
def TVC_forward(input, B, C, L, out_cha,  r):
    TW = []
    THR = [] 
    shape = [out_cha, C, L, L]
    back_shapes = []
    THR.append(input)
    for i in range(r-1):
        if i ==0:
            back_shapes.append((out_cha, C, L, L))
            TW.append(np.ones(shape))
            THR.append((THR[0][:,:,:,None]*THR[0][:,:,None,:]).reshape(B,C,-1))
            out = np.einsum('BCL,OCL -> BO',THR[1], TW[0].reshape(out_cha, C, -1))
        else:
            shape.append(L)
            back_shapes.append((out_cha, C, L, L**(1+i)))
            TW.append(np.ones(shape))
            THR.append((THR[i][:,:,:,None]*THR[0][:,:,None,:]).reshape(B,C,-1))
            out += np.einsum('BCL,OCL -> BO',THR[i+1], TW[i].reshape(out_cha, C, -1))   
    return out,  TW, THR, back_shapes


def Transpose_Matrix(No_conv_dims, r):
    Trans_M =[]
    Total_TM = []
   
    for i in range(r-1):
        for j in range(i+1):
            Padding_dim = np.arange(No_conv_dims).tolist()
            Ori_TM = (np.arange(i+2) + No_conv_dims).tolist()
            chosen = Ori_TM[len(Ori_TM)-j-1]
            Ori_TM.remove(chosen) 
            Ori_TM.insert(0,chosen)
            Ori_TM = Padding_dim + Ori_TM
            Trans_M.append(Ori_TM)
        Total_TM.append(Trans_M)
        Trans_M = []
    return Total_TM  

@profile
def TVC_backward(gradient, TW, THR, back_shapes, Total_TM, r):
    
    for i in range(r-1):
        if i ==0:
            W_back = np.zeros(back_shapes[i]) 
            W_back += TW[0].reshape(back_shapes[0])
            W_back += TW[0].transpose(Total_TM[0][0]).reshape(back_shapes[0])
            grad = np.einsum('BO,OCIL,BCL -> BCI',gradient, W_back, THR[0])
        else:
            W_back =  np.zeros(back_shapes[i]) 
            W_back += TW[i].reshape(back_shapes[i]) 
            for j in range(i+1):
                W_back += TW[i].transpose(Total_TM[i][j]).reshape(back_shapes[i])
            grad += np.einsum('BO,OCIL,BCL -> BCI',gradient, W_back, THR[i])
    return grad


# ####################################      speed up comparsion   ###############################
L_min = 9
L_max = 25  # input size
B = 10 # Batch size
C = 10 # input channel
O = 10 # out channel

# for i in range(L_max-L_min):
#     L = i+ L_min
#     input = (np.arange(B*C*L)[:,None] + 1).reshape(B,C,L)  # Shape of input B,C,L
    


#     PM2_full = PM_Full_creation(2, L, TRM2)
#     PM3_full = PM_Full_creation(3, L, TRM3)
#     PM4_full = PM_Full_creation(4, L, TRM4)
    
#     PCMs_2 = [PM2_full]
#     PCMs_3 = PCM_creation(PM2_full, PM3_full)
#     PCMs_4 = PCM_creation(PM3_full, PM4_full)
    
#     TPCMs_2 = [PCMs_2]
#     TPCMs_3 =[PCMs_2, PCMs_3]
#     TPCMs_4 =[PCMs_2, PCMs_3, PCMs_4]
    
    

#     out_2_E, TW_2_E, THR_2_E, shapes_2_E = EVC_forward(input, B,C,L, O, TPCMs_2)
#     out_3_E, TW_3_E, THR_3_E, shapes_3_E = EVC_forward(input, B,C,L, O, TPCMs_3)
#     out_4_E, TW_4_E, THR_4_E, shapes_4_E = EVC_forward(input, B,C,L, O, TPCMs_4)
    
#     out_2_E = np.ones(out_2_E.shape)
#     out_3_E = np.ones(out_3_E.shape)
#     out_4_E = np.ones(out_4_E.shape)
    
#     Trans_M_2 = Transpose_Matrix(2, 2)
#     Trans_M_3 = Transpose_Matrix(2, 3)
#     Trans_M_4 = Transpose_Matrix(2, 4)
    
#     out_2_T, TW_2_T, THR_2_T, shapes_2_T = TVC_forward(input, B,C,L, O, 2)
#     out_3_T, TW_3_T, THR_3_T, shapes_3_T = TVC_forward(input, B,C,L, O, 3)
#     out_4_T, TW_4_T, THR_4_T, shapes_4_T = TVC_forward(input, B,C,L, O, 4)
    
#     out_2_T = np.ones(out_2_T.shape)
#     out_3_T = np.ones(out_3_T.shape)
#     out_4_T = np.ones(out_4_T.shape)
#     ########### forward and backward of EVC and TVC ###############################
    
#         ####################  second order EVC ####################
#     EVC_forward_timer_2 = Timer(lambda:EVC_forward(input, B,C,L, O, TPCMs_2))
#     EVC_forward_execution_time_2 = EVC_forward_timer_2.timeit(number=10)
    
    
#     EVC_backward_timer_2 = Timer(lambda:EVC_backward(out_2_E, TW_2_E, THR_2_E, shapes_2_E, TPCMs_2))
#     EVC_backward_execution_time_2 = EVC_backward_timer_2.timeit(number=10)
    
   
#         ####################  second order TVC ####################
    
    
#     TVC_forward_timer_2 = Timer(lambda:TVC_forward(input, B,C,L, O, 2))
#     TVC_forward_execution_time_2 = TVC_forward_timer_2.timeit(number=10)
    
#     TVC_backward_timer_2 = Timer(lambda:TVC_backward(out_2_T, TW_2_T, THR_2_T, shapes_2_T, Trans_M_2, 2))
#     TVC_backward_execution_time_2 = TVC_backward_timer_2.timeit(number=10)
    
#     Second_order_forward_speedup.append(TVC_forward_execution_time_2/EVC_forward_execution_time_2)
#     Second_order_backward_speedup.append(TVC_backward_execution_time_2/EVC_backward_execution_time_2)
    
    
#         ####################  Third order EVC ####################
    
#     EVC_forward_timer_3 = Timer(lambda:EVC_forward(input, B,C,L, O, TPCMs_3))
#     EVC_forward_execution_time_3 = EVC_forward_timer_3.timeit(number=10)
    
#     EVC_backward_timer_3 = Timer(lambda:EVC_backward(out_3_E, TW_3_E, THR_3_E, shapes_3_E, TPCMs_3))
#     EVC_backward_execution_time_3 = EVC_backward_timer_3.timeit(number=10)
    
#     #     ####################  Third order TVC ####################
    
#     TVC_forward_timer_3 = Timer(lambda:TVC_forward(input, B,C,L, O, 3))
#     TVC_forward_execution_time_3 = TVC_forward_timer_3.timeit(number=10)
    
#     TVC_backward_timer_3 = Timer(lambda:TVC_backward(out_3_T, TW_3_T, THR_3_T, shapes_3_T, Trans_M_3, 3))
#     TVC_backward_execution_time_3 = TVC_backward_timer_3.timeit(number=10)
    
#     Third_order_forward_speedup.append(TVC_forward_execution_time_3/EVC_forward_execution_time_3)
#     Third_order_backward_speedup.append(TVC_backward_execution_time_3/EVC_backward_execution_time_3)
    
    
#     #     ####################  Fourth order EVC ####################
#     EVC_forward_timer_4 = Timer(lambda:EVC_forward(input, B,C,L, O, TPCMs_4))
#     EVC_forward_execution_time_4 = EVC_forward_timer_4.timeit(number=10)
    
#     EVC_backward_timer_4 = Timer(lambda:EVC_backward(out_4_E, TW_4_E, THR_4_E, shapes_4_E, TPCMs_4))
#     EVC_backward_execution_time_4 = EVC_backward_timer_4.timeit(number=10)
    
    
   
#     #     ####################  Fourth order TVC ####################
#     TVC_forward_timer_4 = Timer(lambda:TVC_forward(input, B,C,L, O, 4))
#     TVC_forward_execution_time_4 = TVC_forward_timer_4.timeit(number=10)
    
    
#     TVC_backward_timer_4 = Timer(lambda:TVC_backward(out_4_T, TW_4_T, THR_4_T, shapes_4_T, Trans_M_4, 4))
#     TVC_backward_execution_time_4 = TVC_backward_timer_4.timeit(number=10)
    
#     Fourth_order_forward_speedup.append(TVC_forward_execution_time_4/EVC_forward_execution_time_4)
#     Fourth_order_backward_speedup.append(TVC_backward_execution_time_4/EVC_backward_execution_time_4)
    
# Second_order_forward_speedup = np.vstack(Second_order_forward_speedup)    
# Third_order_forward_speedup = np.vstack(Third_order_forward_speedup)    
# Fourth_order_forward_speedup = np.vstack(Fourth_order_forward_speedup)  

# Second_order_backward_speedup = np.vstack(Second_order_backward_speedup)    
# Third_order_backward_speedup = np.vstack(Third_order_backward_speedup)    
# Fourth_order_backward_speedup = np.vstack(Fourth_order_backward_speedup)  

# savemat('Second_order_forward_speedup.mat', {'a':Second_order_forward_speedup})
# savemat('Third_order_forward_speedup.mat', {'a':Third_order_forward_speedup}) 
# savemat('Fourth_order_forward_speedup.mat', {'a':Fourth_order_forward_speedup})  


# savemat('Second_order_backward_speedup.mat', {'a':Second_order_backward_speedup})
# savemat('Third_order_backward_speedup.mat', {'a':Third_order_backward_speedup}) 
# savemat('Fourth_order_backward_speedup.mat', {'a':Fourth_order_backward_speedup})  

# ####################################      Space consuming comparsion   ###############################
# L_min = 9
# L = 25  # input size
# B = 1 # Batch size
# C = 1000 # input channel
# O = 1 # out channel

# PM2_full = PM_Full_creation(2, L, TRM2)
# PM3_full = PM_Full_creation(3, L, TRM3)
# PM4_full = PM_Full_creation(4, L, TRM4)

# PCMs_2 = [PM2_full]
# PCMs_3 = PCM_creation(PM2_full, PM3_full)
# PCMs_4 = PCM_creation(PM3_full, PM4_full)

# TPCMs_2 = [PCMs_2]
# TPCMs_3 =[PCMs_2, PCMs_3]
# TPCMs_4 =[PCMs_2, PCMs_3, PCMs_4]

# input = (np.arange(B*C*L)[:,None] + 1).reshape(B,C,L)  # Shape of input B,C,L

# out_2_E, TW_2_E, THR_2_E, shapes_2_E = EVC_forward(input, B,C,L, O, TPCMs_2) 
# out_2_T, TW_2_T, THR_2_T, shapes_2_T = TVC_forward(input, B,C,L, O, 2)   

# out_3_E, TW_3_E, THR_3_E, shapes_3_E = EVC_forward(input, B,C,L, O, TPCMs_3)
# out_3_T, TW_3_T, THR_3_T, shapes_3_T = TVC_forward(input, B,C,L, O, 3)

# out_4_E, TW_4_E, THR_4_E, shapes_4_E = EVC_forward(input, B,C,L, O, TPCMs_4)
# out_4_T, TW_4_T, THR_4_T, shapes_4_T = TVC_forward(input, B,C,L, O, 4)

# out_2_E = np.ones(out_2_E.shape)
# out_3_E = np.ones(out_3_E.shape)
# out_4_E = np.ones(out_4_E.shape)

# Trans_M_2 = Transpose_Matrix(2, 2)
# Trans_M_3 = Transpose_Matrix(2, 3)
# Trans_M_4 = Transpose_Matrix(2, 4)

# out_2_T = np.ones(out_2_T.shape)
# out_3_T = np.ones(out_3_T.shape)
# out_4_T = np.ones(out_4_T.shape)

# grad_2_E = EVC_backward(out_2_E, TW_2_E, THR_2_E, shapes_2_E, TPCMs_2)
# grad_2_T = TVC_backward(out_2_T, TW_2_T, THR_2_T, shapes_2_T, Trans_M_2, 2)   

# grad_3_E = EVC_backward(out_3_E, TW_3_E, THR_3_E, shapes_3_E, TPCMs_3)
# grad_3_T = TVC_backward(out_3_T, TW_3_T, THR_3_T, shapes_3_T, Trans_M_3, 3)

# grad_4_E = EVC_backward(out_4_E, TW_4_E, THR_4_E, shapes_4_E, TPCMs_4)
# grad_4_T = TVC_backward(out_4_T, TW_4_T, THR_4_T, shapes_4_T, Trans_M_4, 4)  
    
