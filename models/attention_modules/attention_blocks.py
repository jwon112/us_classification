import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class CustomSEBlock(nn.Module):
    """Custom Squeeze-and-Excitation Block for channel attention (GELU version)"""
    def __init__(self, in_channels, reduction=16):
        super(CustomSEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.gelu(self.fc1(w))
        w = self.gelu(self.fc2(w))
        return x * w

class SEBlock(nn.Module):
    """Standard Squeeze-and-Excitation Block (RepVGG compatible)"""
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class ChannelAttention(nn.Module):
    """Channel Attention Module combining average and max pooling"""
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

class EnhancedSpatialAttention(nn.Module):
    """Enhanced Spatial Attention Module using depthwise separable convolution"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(2, 1, kernel_size, padding=kernel_size // 2, dilation=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SSEBlock(nn.Module):
    """Spatial Squeeze and Channel Excitation Block"""
    def __init__(self, channels):
        super(SSEBlock, self).__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1)  # For spatial squeeze
        self.conv_channel = nn.Conv2d(channels, channels, kernel_size=1)  # For channel excitation
        self.sigmoid = nn.Sigmoid()
        self.swish = Swish()

    def forward(self, x):
        # Spatial Squeeze: compress spatial dimensions to a single channel
        spatial_squeeze = self.conv_spatial(x)
        spatial_squeeze = self.sigmoid(spatial_squeeze)

        # Channel Excitation: emphasize important channels
        channel_excitation = self.conv_channel(x)
        channel_excitation = self.sigmoid(channel_excitation)

        # Element-wise multiplication of spatial and channel attention
        out = spatial_squeeze * channel_excitation * x
        return out

class FEEBlock(nn.Module):
    """Feature Enhancement and Extraction Block"""
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
    """Deep Feature Mining Block"""
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
    """Depthwise Separable Convolution for efficient computation"""
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    """Basic Residual Block"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(x)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class R2Block(nn.Module):
    """Recursive Residual Block"""
    def __init__(self, in_channels, out_channels, recursion=2):
        super(R2Block, self).__init__()
        self.recursion = recursion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        for _ in range(self.recursion):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += residual  # Add the residual connection
        return F.relu(x)

# Attention Module Factory
class AttentionFactory:
    """Factory class for creating attention modules"""
    
    @staticmethod
    def get_attention(attention_type, channels, **kwargs):
        """Get attention module by type"""
        if attention_type == 'se':
            return SEBlock(channels, **kwargs)
        elif attention_type == 'custom_se':
            return CustomSEBlock(channels, **kwargs)
        elif attention_type == 'channel':
            return ChannelAttention(channels, **kwargs)
        elif attention_type == 'spatial':
            return EnhancedSpatialAttention(**kwargs)
        elif attention_type == 'sse':
            return SSEBlock(channels)
        elif attention_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
    
    @staticmethod
    def get_residual_block(block_type, in_channels, out_channels, **kwargs):
        """Get residual block by type"""
        if block_type == 'basic':
            return BasicBlock(in_channels, out_channels, **kwargs)
        elif block_type == 'r2':
            return R2Block(in_channels, out_channels, **kwargs)
        elif block_type == 'none':
            return None
        else:
            raise ValueError(f"Unknown residual block type: {block_type}")
