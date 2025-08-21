import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SSEBlock(nn.Module):
    def __init__(self, channels):
        super(SSEBlock, self).__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1)
        self.conv_channel = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.swish = Swish()

    def forward(self, x):
        spatial_squeeze = self.conv_spatial(x)
        spatial_squeeze = self.sigmoid(spatial_squeeze)
        channel_excitation = self.conv_channel(x)
        channel_excitation = self.sigmoid(channel_excitation)
        out = spatial_squeeze * channel_excitation * x
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.swish = Swish()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class FEEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        return x

class DFMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFMBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        return x

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class R2Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(R2Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        return out

class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))

class EnhancedShuffleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # ShuffleNetV2 백본 사용
        shufflenet = models.shufflenet_v2_x1_0(weights=None)
        
        # ShuffleNetV2의 구조: conv1 -> stage2 -> stage3 -> stage4 -> conv5
        self.conv1 = shufflenet.conv1
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 = shufflenet.stage4
        self.conv5 = shufflenet.conv5
        
        # ShuffleNetV2의 실제 출력 채널 수 감지
        # 테스트 입력으로 채널 수 확인
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.conv1(test_input)
            test_output = self.stage2(test_output)
            test_output = self.stage3(test_output)
            test_output = self.stage4(test_output)
            test_output = self.conv5(test_output)
            actual_channels = test_output.shape[1]
            print(f"ShuffleNet actual output channels: {actual_channels}")
        
        # 1x1 convolution으로 채널 수를 1024로 조정
        self.conv1x1 = nn.Conv2d(actual_channels, 1024, kernel_size=1)
        
        # 기존과 동일한 attention 및 custom layers
        self.r2_block = R2Block(1024, 1024)
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # ShuffleNetV2 feature extraction
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        
        x = self.conv1x1(x)  # 채널 수 조정 (필요시)
        
        x = self.se_block(x)  # Squeeze and Excitation block
        x = self.channel_attention(x) * x  # Channel attention
        x = self.spatial_attention(x) * x  # Spatial attention
        x = self.residual_block(x)  # Residual block
        
        x = F.dropout(x, p=0.3, training=self.training)  # Dropout
        
        x = self.se_block(x)
        x = self.channel_attention(x) * x  # Channel attention
        x = self.spatial_attention(x) * x  # Spatial attention
        x = self.residual_block(x)  # Residual block
        
        x = self.pooling(x)  # Adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        x = self.fc(x)  # Output layer
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))
