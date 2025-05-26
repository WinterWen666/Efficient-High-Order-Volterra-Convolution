import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from GCHOC import Second_GCHOC




def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

    
class SE(nn.Module):

    def __init__(self, channels, reduction):
        super(SE, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
      
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
    
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.att = SE(inplanes,16)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # out = self.att(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
       
class Vol_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  downsample=None):
        super(Vol_Bottleneck, self).__init__()
        # self.att = SE(inplanes,16)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.act = nn.Tanh()  
        self.conv3 = Second_GCHOC(planes, planes * Bottleneck.expansion,4,1,1)   

       
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # out = self.att(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out        


class Fused_block(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Fused_block, self).__init__()
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
            )
        self.conv1 = conv3x3(inplanes, planes *4, stride)
        
        self.bn1 = nn.BatchNorm2d(inplanes)
       
        self.conv2 = nn.Conv2d(planes *4, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes *4)
        self.relu = nn.ReLU()

        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
       
        out = self.conv2(out)

        
        out += residual
        
        return out  
        
class Double_Vol_Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Double_Vol_Bottleneck, self).__init__()
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != planes * Vol_Bottleneck.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * Vol_Bottleneck.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )
         
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.act =nn.Tanh()  
        self.conv1 = Second_GCHOC(inplanes, planes * Vol_Bottleneck.expansion,4,1,1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = Second_GCHOC(planes, planes * Vol_Bottleneck.expansion,4,1,1)   
        self.relu = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(planes * Vol_Bottleneck.expansion)
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        
        out += residual
        
        return out



class ResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes):
        super(ResNet, self).__init__()        
        self.dataset = dataset
        if self.dataset.startswith('cifar'):
            self.inplanes = 16
            print(bottleneck)
            if bottleneck == True:
                n = int((depth - 2) / 9)
                block = Bottleneck
            else:
                n = int((depth - 2) / 6)
                block = BasicBlock

            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, n)
            self.layer2 = self._make_layer(block, 32, n, stride=2)
            self.layer3 = self._make_layer(block, 64, n, stride=2) 
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        elif dataset == 'imagenet':
            blocks ={ 1: BasicBlock,2: Fused_block, 3: Bottleneck, 4: Vol_Bottleneck, 5: Double_Vol_Bottleneck} 
            layers ={18: [2, 2, 2, 2], 34: [3, 4, 6, 3], 50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 110: [3, 4, 23, 6], 152: [3, 8, 36, 3], 200: [3, 24, 36, 3]}
            assert layers[depth], 'invalid detph for ResNet (depth should be one of 18, 34, 50, 101, 152, and 200)'

            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(blocks[3], 64, layers[depth][0])
            self.layer2 = self._make_layer(blocks[3], 128, layers[depth][1], stride=2)
            self.layer3 = self._make_layer(blocks[3], 256, layers[depth][2], stride=2)
            self.layer4 = self._make_layer(blocks[4], 512, layers[depth][3], stride=2)
            ## Used for fuesd_block and Double_Vol_Bottleneck
            # self.vol_conv = Vol_CNN(512*blocks[2].expansion, 512*blocks[2].expansion*4,4,1,1)
            # self.bn_last = nn.BatchNorm2d(512*blocks[4].expansion)
            self.avgpool = nn.AvgPool2d(7) 
            self.fc = nn.Linear(512*blocks[4].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, layers, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        Blocks = []
        Blocks.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, layers):
            Blocks.append(block(self.inplanes, planes))

        ## Used for fuesd_block and Double_Vol_Bottleneck
        # Blocks = []
        # for i in range(layers):
        #     Blocks.append(blocks(self.inplanes, planes,stride))
        #     self.inplanes = planes * blocks.expansion
        #     stride = 1              
        return nn.Sequential(*Blocks)

    def forward(self, x):
        if self.dataset == 'cifar10' or self.dataset == 'cifar100':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        elif self.dataset == 'imagenet':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            # x = self.vol_conv(x) 
            #x = self.bn_last(x)
            
            x = self.relu(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
    
        return x