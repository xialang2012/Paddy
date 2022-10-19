import torch
import torch.nn as nn
import torch.nn.functional as F

channels = [32, 64, 128, 256]

def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
class FullConfusionUnit(nn.Module):

    def __init__(self, in_channels, out_channels, with_conv_shortcut=False, halfChannels=None):

        self.halfChannels = halfChannels
        self.with_conv_shortcut = with_conv_shortcut

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        out = self.conv(x)
        out = self.conv(out)

        if self.with_conv_shortcut:
            residual = self.conv(x)
            out = out.add(residual)
        else:
            out = out.add(x)

        if self.halfChannels:
            out_half = F.interpolate(out, scale_factor=1.0/2, mode='bilinear')
        else:
            out_half = None

        return out, out_half

class BasicBlock(nn.modules):
    
    def __init__(self, in_channels, out_channels, with_conv_shortcut=False) -> None:
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)            
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)            
        )

        self.relu = nn.ReLU(inplace=True)

        self.with_conv_shortcut = with_conv_shortcut

    def forward(self, x):

        out = self.conv(x)

        out = self.conv2(out)

        if self.with_conv_shortcut:
            residual = self.conv3(x)
            out += residual
        else:
            out += x

        out = self.relu(out)
        return out

class FRNet(nn.modules):   

    def __init__(self, in_channels, out_channels, classes=1) -> None:
        super().__init__()

        self.inHeadChannels = channels[3]

        self.inconv = nn.Sequential(
            nn.Conv2d(in_channels, self.inHeadChannels, kernel_size=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inHeadChannels),
            nn.ReLU(inplace=True)
        )

        # 1*1 convs for stage 1 and 2
        self.conv1_s1 = conv1x1(self.inHeadChannels, channels[0])
        self.conv1_s2 = conv1x1(self.inHeadChannels, channels[0])

        # 1*1 convs for stage 3
        self.conv1_s3_fs2_l2_f = conv1x1(channels[1], channels[0])
        self.conv1_s3_fs3_l2_f = conv1x1(channels[1], channels[1])
        self.conv1_s3_fs3_l3_f = conv1x1(channels[2], channels[2])
        self.conv1_s3_fs2_l2_f_d = conv1x1(channels[1], channels[2])
        self.conv1_s3_fs1_4f = conv1x1(self.inHeadChannels, channels[2])

        # 1*1 convs for stage 4
        self.conv1_s4_fs3_l2_f = conv1x1(channels[1], channels[0])
        self.conv1_s4_fs3_l3_f = conv1x1(channels[2], channels[0])
        self.conv1_s4_fs4_l2_f = conv1x1(channels[1], channels[1])
        self.conv1_s4_fs3_l3_f_1 = conv1x1(channels[2], channels[1])

        self.conv1_s4_fs4_l3_f = conv1x1(channels[2], channels[2])
        self.conv1_s4_fs4_l4_f = conv1x1(channels[3], channels[3])
        self.conv1_s4_fs3_l3_f_2 = conv1x1(channels[2], channels[3])
        self.conv1_s4_fs2_l2_f = conv1x1(channels[1], channels[3])
        self.conv1_s4_fs1_8f = conv1x1(self.inHeadChannels, channels[3])
    
        # basic block for stage 1 to 4
        self.BasicBlock_s1 = BasicBlock(channels[0], channels[0])
        self.BasicBlock_s2 = BasicBlock(channels[1], channels[1])
        self.BasicBlock_s3 = BasicBlock(channels[2], channels[2])
        self.BasicBlock_s4 = BasicBlock(channels[3], channels[3])

        # confusion unit for stage 2, 3, 4
        # s3_l1_f, channels[0]
        self.funit_s3_l1_f = FullConfusionUnit(channels[0], channels[0])  
        # s3_l2_f, channels[1]
        self.funit_s3_l2_f = FullConfusionUnit(channels[1], channels[1])
        # s4_l1_f, channels[0]
        self.funit_s4_l1_f = FullConfusionUnit(channels[0], channels[0])
        # s4_l2_f, channels[1]
        self.funit_s4_l2_f = FullConfusionUnit(channels[1], channels[1])
        # s4_l3_f, channels[2]
        self.funit_s4_l3_f = FullConfusionUnit(channels[2], channels[2])
        # s4_l4_f, channels[3]
        self.funit_s4_l4_f = FullConfusionUnit(channels[3], channels[3], halfChannels=None)

        # final class
        self.conFinal = conv1x1(sum(channels), 1)
    
    def forward(self, x):

        s1_f = self.inconv(x)

        s1_2f = F.interpolate(s1_f, scale_factor=1.0/2, mode='bilinear')
        s1_4f = F.interpolate(s1_f, scale_factor=1.0/4, mode='bilinear')
        s1_8f = F.interpolate(s1_f, scale_factor=1.0/8, mode='bilinear')

        # stage2, full & aspp layer 1
        s1_f = self.conv1_s1(s1_f)
        s2_l1_f = self.BasicBlock_s1(s1_f)
        s2_l1_f = self.BasicBlock_s1(s2_l1_f) # L1
        s2_l1_f = self.BasicBlock_s1(s2_l1_f) # L1

        s2_l2_f = self.BasicBlock_s2(s1_2f) # L2
        s2_l2_f = self.BasicBlock_s2(s2_l2_f) # L2
        s2_l2_f = self.BasicBlock_s2(s2_l2_f) # L2

        # stage 3
        # s3_l1_f = Conv2D(channels[0], 1, use_bias=False, kernel_initializer='he_normal')(s2_l2_f)
        # s3_l1_f = BatchNormalization(axis=3)(s3_l1_f)
        s3_l1_f = self.conv1_s3_fs2_l2_f(s2_l2_f)
        s3_l1_f = F.interpolate(s3_l1_f, scale_factor=2, mode='bilinear')
        s3_l1_f += s2_l1_f
        s3_l1_f, s3_l2_f = self.funit_s3_l1_f(s3_l1_f)
        s3_l1_f = self.BasicBlock_s1(s3_l1_f)      # L1
        s3_l1_f = self.BasicBlock_s1(s3_l1_f)      # L1

        s3_l2_f = self.conv1_s3_fs3_l2_f(s3_l2_f)
        s3_l2_f += s2_l2_f
        s3_l2_f, s3_l3_f = self.funit_s3_l2_f(s3_l2_f)
        s3_l2_f = self.BasicBlock_s2(s3_l2_f)      # L2
        s3_l2_f = self.BasicBlock_s2(s3_l2_f)      # L2

        s3_l3_f = self.conv1_s3_fs3_l3_f(s3_l3_f)
        s2_l2_f_d = F.interpolate(s2_l2_f, scale_factor=0.5, mode='bilinear')
        s2_l2_f_d = self.conv1_s3_fs2_l2_f_d(s2_l2_f_d)
        s1_4f_c = self.conv1_s3_fs1_4f(s1_4f)
        s3_l3_f = s3_l3_f.add(s2_l2_f_d).add(s1_4f_c)
        s3_l3_f = self.BasicBlock_s3(s3_l3_f)      # L3 
        s3_l3_f = self.BasicBlock_s3(s3_l3_f)      # L3    

        # stage 4
        s4_l1_f_d = self.conv1_s4_fs3_l2_f(s3_l2_f)
        s4_l1_f_d = F.interpolate(s4_l1_f_d, scale_factor=2, mode='bilinear')

        s4_l1_f_2up = self.conv1_s4_fs3_l3_f(s3_l3_f)
        s4_l1_f_2up = F.interpolate(s4_l1_f_2up, scale_factor=4, mode='bilinear')

        s4_l1_f = s3_l1_f.add(s4_l1_f_d).add(s4_l1_f_2up)
        s4_l1_f, s4_l2_f = self.funit_s4_l1_f(s4_l1_f)
        s4_l1_f = self.BasicBlock_s1(s4_l1_f)      # L1
        s4_l1_f = self.BasicBlock_s1(s4_l1_f)      # L1

        #s4_l2_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l2_f)    # downsample
        s4_l2_f_d = self.conv1_s4_fs4_l2_f(s4_l2_f)   # channels
        s4_l2_f_up = self.conv1_s4_fs3_l3_f_1(s3_l3_f)
        s4_l2_f_up = F.interpolate(s4_l2_f_up, scale_factor=2, mode='bilinear')
        s4_l2_f = s4_l2_f_d.add(s4_l2_f_up).add(s3_l2_f)
        s4_l2_f, s4_l3_f = self.funit_s4_l2_f(s4_l2_f)
        s4_l2_f = self.BasicBlock_s2(s4_l2_f)      # L2
        s4_l2_f = self.BasicBlock_s2(s4_l2_f)      # L2


        #s4_l3_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l3_f)    # downsample
        s4_l3_f_d = self.conv1_s4_fs4_l3_f(s4_l3_f)   # channels  
        s4_l3_f = s4_l3_f_d.add(s3_l3_f)
        s4_l3_f, s4_l4_f = self.funit_s4_l3_f(s4_l3_f)
        s4_l3_f = self.BasicBlock_s3(s4_l3_f)      # L3
        s4_l3_f = self.BasicBlock_s3(s4_l3_f)      # L3


        #s4_l4_f_d = MaxPooling2D(pool_size=(2, 2))(s4_l4_f)    
        s4_l4_f_d = self.conv1_s4_fs4_l4_f(s4_l4_f)   # channels
        
        s3_l3_f_d = self.conv1_s4_fs3_l3_f_2 (s3_l3_f)   # channels
        s3_l3_f_d = F.interpolate(s3_l3_f_d, scale_factor=0.5, mode='bilinear')

        s2_l2_f_2d = self.conv1_s4_fs2_l2_f(s2_l2_f)   # channels
        s2_l2_f_2d = F.interpolate(s2_l2_f_2d, scale_factor=0.25, mode='bilinear')

        s1_8f = self.conv1_s4_fs1_8f(s1_8f)

        s4_l4_f = s2_l2_f_2d.add(s4_l4_f_d).add(s1_8f).add(s3_l3_f_d)
        s4_l4_f, s4_l5_f = self.funit_s4_l4_f(s4_l4_f)
        s4_l4_f = self.BasicBlock_s4(s4_l4_f)      # L4
        s4_l4_f = self.BasicBlock_s4(s4_l4_f)      # L4

        # upsampling
        s4_l2_f = F.interpolate(s4_l2_f, scale_factor=2, mode='bilinear')
        s4_l3_f = F.interpolate(s4_l3_f, scale_factor=4, mode='bilinear')
        s4_l4_f = F.interpolate(s4_l4_f, scale_factor=8, mode='bilinear')

        x = torch.cat([s4_l1_f, s4_l2_f, s4_l3_f, s4_l4_f], 1)

        return self.conFinal(x)
        # if classes == 1:
        #     x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        #     x = BatchNormalization(axis=3)(x)
        #     out = Activation('sigmoid', name='Classification')(x)
        # else:
        #     x = Conv2D(classes, 1, use_bias=False, kernel_initializer='he_normal')(x)
        #     x = BatchNormalization(axis=3)(x)
        #     out = Activation('softmax', name='Classification')(x)