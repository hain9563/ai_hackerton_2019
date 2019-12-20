import torch
from torch import nn
from torch.nn import functional

# Down SIze : n => (n+1)//2
# n => 1 까지 줄이는 데 필요한 DownSampling 갯수 :
#     2^(k-1)  <  n  <=  2^k  =>  k개
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DownSampling, self).__init__()
        self.stride = stride
        if(self.stride == 2):
            self.pad = nn.ZeroPad2d((0,1,0,1))
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
        else:
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        if(self.stride == 2):
            x = self.pad(x)
        x = self.down(x)
        x = self.bn(x)
        return x


class EggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EggBlock, self).__init__()

        self.down_sample = DownSampling(in_channels, out_channels, stride)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)    
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
            

    def forward(self, x):
        shortcut = self.down_sample(x)

        out = self.conv1(x)   
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += shortcut
        out = self.relu2(out)
        return out


class EggNet(nn.Module):
    def __init__(self, num_layers, block, num_classes=4):
        super(EggNet, self).__init__()   	
        self.in_channels = 16
        self.stride = 1
        self.num_layers = num_layers
        self.hidden_node = 16
        self.drop_rate = 0.3

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.layer1 = self.get_layers(block, addrate=1.2)
        self.layer2 = self.get_layers(block, addrate=0.3)
        self.layer3 = self.get_layers(block, addrate=0.2)

        self.out_channels = int(round(self.out_channels))
        self.class_out1 = nn.Conv2d(self.out_channels, self.hidden_node, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(self.hidden_node)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.drop_rate)
        self.class_out2 = nn.Conv2d(self.hidden_node, num_classes, kernel_size=1, stride=1)
        #self.globalavg = nn.AdaptiveAvgPool2d((1,1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def get_layers(self, block, addrate):
        layers_list = []
        for _ in range(self.num_layers):
            self.out_channels = self.in_channels * (1+addrate)
            layers_list.append(block(int(round(self.in_channels)), int(round(self.out_channels)), stride=self.stride))
            self.in_channels = self.out_channels
            if(self.stride == 1):
                self.stride = 2

        return nn.Sequential(*layers_list)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.class_out1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.class_out2(x)
        #x = self.globalavg(x)

        x = x.view(x.size(0), -1)
        return x


def getEggNet(num_classes):
	block = EggBlock
	model = EggNet(num_layers=3, block=block, num_classes=num_classes)
	return model


