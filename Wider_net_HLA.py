

import torch

import numpy as np
import torch.nn as nn
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from Efficient_High_order_conv import PM_creation, PCM_creation, HO_conv
from torchvision.transforms.autoaugment import AutoAugmentPolicy

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


train_transform = transforms.Compose([
   
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    # transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    # Cutout(1, 16),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

# Load CIFAR-100 dataset with data augmentation
train_dataset = CIFAR100(root='./data', train=True, download=True, transform=train_transform)
test_dataset = CIFAR100(root='./data', train=False, download=True, transform=test_transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)  
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)  
                                           

num_classes = 100
k = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class squeeze(nn.Module):
    def __init__(self, in_chan, reduce):
        super(squeeze, self).__init__()
       
       
        self.squ = nn.Sequential( 
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_chan, in_chan//reduce,1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_chan//reduce,in_chan, 1),
                                    nn.Sigmoid()
                                    )
    def forward(self, input):
        x = self.squ(input)*input
        return x

class Volconv(nn.Module):
    def __init__(self, in_chan,out_chan, kernel_size, stride, padding, groups):
        super(Volconv, self).__init__()
        input_size = int(kernel_size[0]*kernel_size[1]) 
        # PM1_Full = torch.arange(input_size)[:,None]
        TRM2 = [[2],[1,1]] 
        PM2_Full = PM_creation(2, input_size, TRM2)
        # TRM3 = [[3],[1,2],[2, 1], [ 1, 1, 1]]  # Third order Toe 
        # PM3_Full = PM_creation(3, input_size, TPE3)
        TPCMs_r = [[PM2_Full]]
        # PCMs_r = PCM_creation(PM2_Full, PM3_Full)
        
        self.Second_conv = nn.Sequential( 
                                   
                                   HO_conv(in_chan,  out_chan,
                                              kernel_size  , 
                                              stride  , padding, groups, TPCMs_r
                                              ), )
       
        
        self.conv = nn.Sequential(
                                   nn.Conv2d(in_chan, out_chan, kernel_size, 
                                              stride , padding, groups = groups, bias = False))
        
    def forward(self, input):
        x = self.Second_conv(input)+self.conv(input)
        
        return x 
class HLA(nn.Module):
    def __init__(self, in_chan, kernel_size, pooling_size, padding,  in_size, reduce):
        super(HLA, self).__init__()
        up_size1 = in_size[0]//pooling_size[0]
        up_size2 = in_size[1]//pooling_size[1]
       
        self.Vol1= nn.Sequential(   nn.BatchNorm2d( in_chan),
                                    nn.ReLU(),
                                    
                                    nn.AdaptiveAvgPool2d((pooling_size[0], pooling_size[1])),
                                  
                                    nn.Conv2d(in_chan, in_chan//reduce, kernel_size=1, 
                                                stride=1 , padding = 0, groups = 1),
                                    nn.BatchNorm2d( in_chan//reduce),
                                    
                                    Volconv(in_chan//reduce,in_chan//reduce, kernel_size, (1, 1), padding, 1),
                                  
                                    nn.ReLU(),
                                    nn.BatchNorm2d( in_chan//reduce),
                                    
                                    nn.Conv2d(in_chan//reduce, in_chan, kernel_size=1, 
                                                stride=1 , padding = 0, groups = 1),
                                    
                                    nn.Sigmoid(),
                                    
                                    )
        self.sq = nn.Sequential( 
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Conv2d(in_chan, in_chan//reduce, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_chan//reduce, in_chan, 1),
                                    nn.Sigmoid()
                                    )
       
        
        self.up = nn.Upsample(scale_factor= (up_size1,up_size2) )
       
       
                                  
    def forward(self, input):
        x1 = self.Vol1(input)
        x2 = self.sq(input)
        x3 = torch.mean(x2, dim=1, keepdim= True)
        x2 = x3 - nn.ReLU()(x3 - x2)
        x = x2 + x1 - x2*x1
        x = self.up(x)*input        
        return x 

class Branch(nn.Module):
    def __init__(self, 
                  in_cha, 
                  out_cha,
                  kernel_size,
                  stride,
                  padding,
                  in_size
                  ):   
    
        super(Branch, self).__init__()
        
        self.shortcut = nn.Sequential()
        if in_cha != out_cha or stride != 1:
            self.shortcut = nn.Sequential( nn.Conv2d(in_cha,
                        out_cha,
                        kernel_size=1,  
                        stride= stride  , padding=0,
                      bias = False),)
             
        self.Branch1_1 = nn.Sequential( 
                                  nn.BatchNorm2d( in_cha),
                                  nn.ReLU(), 
                                  
                                  nn.Conv2d(in_cha, 
                                            out_cha,
                                            kernel_size = 3, 
                                            stride= 1, padding = 1,
                                            bias = False),
                                 
                                  )
                                  
       
        self.Branch1_2 = nn.Sequential(
                                  nn.BatchNorm2d(out_cha),
                                  nn.ReLU(), 
                                
                                  nn.Conv2d( out_cha, 
                                           out_cha,
                                            kernel_size = 3, 
                                            stride= stride, padding = 1 ,
                                            bias = False),
                                
                             )
        
        
        
            
    def forward(self, input):
        x = self.Branch1_1(input)
        x =  self.Branch1_2(x) + self.shortcut(input) 
        return x 
class Wide_net(nn.Module):
    def __init__(self, num_classes):
        super(Wide_net, self).__init__()
           
        self.start_covlayer = nn.Sequential( 
            nn.Conv2d(3, 16, 3, 1, 1, bias= False),
             
           )   
                                                           
        self.layer1 = nn.Sequential(
                                    
                                    Branch(16, 16*k,  3, 1, 1,32),
                                    HLA(16*k,  (3, 3), (2, 2), (1, 1), (32, 32), 16),
                                   
                                    Branch(16*k, 16*k,  3, 1, 1, 32),
                                    HLA(16*k,  (3, 3),(2, 2), (1, 1), (32, 32), 16),
                                    
                                    # Branch(16*k, 16*k,  3, 1, 1, 32),
                                    # HLA(16*k,  (3, 3), (2, 2), (1, 1), (32, 32), 16),
                                    
                                    # Branch(16*k, 16*k,  3, 1, 1, 32),
                                    # HLA(16*k,  (3, 3), (2, 2), (1, 1), (32, 32), 16),
                                  
                                   
                                    
                                   
                                   
                                    )
       
       
        self.layer2 = nn.Sequential(
                                 
                                    Branch(16*k, 32*k,  3, 2, 1,16),
                                    HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                    
                                    Branch(32*k, 32*k,3,1, 1, 16),
                                    HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                    
                                              
                                    # Branch(32*k, 32*k,3,1, 1, 16),
                                    #HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                    
                                    # Branch(32*k, 32*k,3,1, 1, 16),
                                    #HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                    
                                    
                                    
                                    # Branch(32*k, 32*k,3,1, 1, 16),
                                    #HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                   
                                   
                                    # Branch(32*k, 32*k,3,1, 1, 16),
                                    # HLA(32*k,  (3, 3), (2, 2), (1, 1), (16, 16), 16),
                                    
                                    )
                             
       
    
       
                                   
        
        self.layer3 = nn.Sequential(      
                                       
                                        Branch(32*k, 64*k,  3, 2, 1, 8),
                                        squeeze(64*k,16),
                                       
                                        Branch(64*k, 64*k, 3,1, 1, 8),
                                        
                                        squeeze(64*k,16),
                                     
                                        nn.BatchNorm2d(64*k),
                                        
                                        nn.ReLU(),
                                        
                                        nn.AdaptiveAvgPool2d(1) ,
                                       
                                      
                                            )
       
                                     
        self.fc = nn.Sequential(
        
        nn.Linear(64*k, num_classes),
        
        )
        
          
    def forward(self, x):
        out = self.start_covlayer(x) 
        
        out = self.layer1(out)
        
        out = self.layer2(out)
       
        out = self.layer3(out)
      
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out
     
     
model = Wide_net(num_classes).to(device)


total_step = len(train_loader)
num_epochs = 200

  
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), momentum = 0.9, lr= 0.1,  weight_decay = 0.0005, nesterov=True)

scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120,160], gamma= 0.2)

total_params = sum(param.numel() for param in model.parameters()) 

with open('log_WRN_16_8_1.txt', 'w') as f:  
    f.write('Epochs\tTraining Loss\tTraining_acc\tTesting_acc\n')

for epoch in range(num_epochs):
    model.train()
    model.training = True
    train_total = 0
    train_correct = 0
    for i, (images, labels) in enumerate(train_loader): 
      
        # Move tensors to the configured device
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
       
        loss.backward()
        optimizer.step()
    
        _, predicted = torch.max(outputs.data, 1)
        train_total +=labels.size(0)
        train_correct +=(predicted == labels).sum().item()
        
    scheduler1.step()    
    train_acc = 100 * train_correct/train_total
    
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, train_acc: {:.3f} %'
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item(), train_acc))
   
            
    # Validation
    correct = 0
    total = 0
    model.eval()
    model.training = False
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))
        with open('log_WRN_16_8_1.txt', 'a') as f:
            f.write('[{}/{}]\t{:.4f}\t        {:.3f} %\t      {:.2f} %\n'.format(
                epoch+1,num_epochs, loss.item(), train_acc, 100 * correct / total))      
    
                      
