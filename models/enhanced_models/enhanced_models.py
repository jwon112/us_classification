import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm
from ..attention_modules import (
    SEBlock, ChannelAttention, EnhancedSpatialAttention,
    R2Block, BasicBlock
)

class EnhancedResNet(nn.Module):
    """Enhanced ResNet50 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        resnet = models.resnet50(weights=None)
        
        # Use all layers except the last two (adaptive pooling and fc layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 1x1 convolution to reduce channels from 2048 to 1024
        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.conv1x1(x)   # Reduce channels from 2048 to 1024
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedDenseNet(nn.Module):
    """Enhanced DenseNet121 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        densenet = models.densenet121(weights=None)
        self.features = densenet.features
        
        # 1x1 convolution to adjust channels (1024 -> 1024)
        self.conv1x1 = nn.Conv2d(1024, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = self.conv1x1(x)
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedMobileNet(nn.Module):
    """Enhanced MobileNetV2 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        self.features = mobilenet.features
        
        # 1x1 convolution to adjust channels (1280 -> 1024)
        self.conv1x1 = nn.Conv2d(1280, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedEfficientNet(nn.Module):
    """Enhanced EfficientNet-B0 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)
        self.features = efficientnet.features
        
        # 1x1 convolution to adjust channels (1280 -> 1024)
        self.conv1x1 = nn.Conv2d(1280, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedShuffleNet(nn.Module):
    """Enhanced ShuffleNetV2 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        shufflenet = models.shufflenet_v2_x1_0(weights=None)
        
        # Explicitly define ShuffleNetV2 stages (원본과 동일한 순서)
        self.conv1 = shufflenet.conv1
        self.maxpool = shufflenet.maxpool  # maxpool 추가!
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 = shufflenet.stage4
        self.conv5 = shufflenet.conv5
        
        # 원본 ShuffleNet은 이미 1024 채널을 출력하므로 conv1x1 불필요
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Sequential feature extraction for ShuffleNetV2 (원본과 동일한 순서)
        x = self.conv1(x)
        x = self.maxpool(x)  # maxpool 추가!
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        # conv1x1 제거 - 이미 1024 채널
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedConvNeXt(nn.Module):
    """Enhanced ConvNeXt-Tiny with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        convnext = models.convnext_tiny(weights=None)
        self.features = convnext.features
        
        # Dynamic channel detection for conv1x1
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.features(test_input)
            actual_channels = test_output.shape[1]
        
        # 1x1 convolution to adjust channels
        self.conv1x1 = nn.Conv2d(actual_channels, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedResNeXt(nn.Module):
    """Enhanced ResNeXt50 with custom attention and residual modules"""
    def __init__(self, num_classes=3):
        super().__init__()
        resnext = models.resnext50_32x4d(weights=None)
        
        # Use all layers except the last two (adaptive pooling and fc layer)
        self.features = nn.Sequential(*list(resnext.children())[:-2])
        
        # 1x1 convolution to reduce channels from 2048 to 1024
        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)
        
        # Custom enhancement modules
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.r2_block = R2Block(1024, 1024)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)  # Feature extraction
        x = self.conv1x1(x)   # Reduce channels from 2048 to 1024
        
        # First set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EnhancedViT(nn.Module):
    """Enhanced ViT-B/16 with custom attention and residual modules (no spatial attention for 1x1 features)"""
    def __init__(self, num_classes=3):
        super().__init__()
        # ViT backbone
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)  # Remove classifier
        
        # Get ViT output dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.vit(test_input)
            vit_output_dim = test_output.shape[1]
        
        # Custom enhancement modules for ViT features (no spatial attention for 1x1 features)
        self.se_block = SEBlock(vit_output_dim)
        self.channel_attention = ChannelAttention(vit_output_dim)
        # Skip spatial attention for ViT (1x1 features)
        self.residual_block = BasicBlock(vit_output_dim, vit_output_dim)
        self.r2_block = R2Block(vit_output_dim, vit_output_dim)
        
        # Final layers
        self.fc = nn.Linear(vit_output_dim, num_classes)

    def forward(self, x):
        # ViT feature extraction
        x = self.vit(x)  # Shape: [batch_size, vit_output_dim]
        
        # Reshape for 2D operations (ViT outputs 1D features)
        batch_size = x.shape[0]
        feature_dim = x.shape[1]
        # Create a 2D representation for attention modules
        x_2d = x.unsqueeze(-1).unsqueeze(-1)  # [batch_size, feature_dim, 1, 1]
        
        # First set of enhancements (no spatial attention)
        x_2d = self.se_block(x_2d)
        x_2d = self.channel_attention(x_2d) * x_2d
        x_2d = self.residual_block(x_2d)
        
        x_2d = F.dropout(x_2d, p=0.3, training=self.training)
        
        # Second set of enhancements (no spatial attention)
        x_2d = self.se_block(x_2d)
        x_2d = self.channel_attention(x_2d) * x_2d
        x_2d = self.r2_block(x_2d)
        
        # Flatten back to 1D
        x = x_2d.squeeze(-1).squeeze(-1)  # [batch_size, feature_dim]
        x = self.fc(x)
        return x

class EnhancedSwinTransformer(nn.Module):
    """Enhanced Swin-T with custom attention and residual modules (no spatial attention for 1x1 features)"""
    def __init__(self, num_classes=3):
        super().__init__()
        # Swin Transformer backbone
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=0)  # Remove classifier
        
        # Get Swin output dimension
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.swin(test_input)
            swin_output_dim = test_output.shape[1]
        
        # Custom enhancement modules for Swin features (no spatial attention for 1x1 features)
        self.se_block = SEBlock(swin_output_dim)
        self.channel_attention = ChannelAttention(swin_output_dim)
        # Skip spatial attention for Swin (1x1 features)
        self.residual_block = BasicBlock(swin_output_dim, swin_output_dim)
        self.r2_block = R2Block(swin_output_dim, swin_output_dim)
        
        # Final layers
        self.fc = nn.Linear(swin_output_dim, num_classes)

    def forward(self, x):
        # Swin feature extraction
        x = self.swin(x)  # Shape: [batch_size, swin_output_dim]
        
        # Reshape for 2D operations (Swin outputs 1D features)
        batch_size = x.shape[0]
        feature_dim = x.shape[1]
        # Create a 2D representation for attention modules
        x_2d = x.unsqueeze(-1).unsqueeze(-1)  # [batch_size, feature_dim, 1, 1]
        
        # First set of enhancements (no spatial attention)
        x_2d = self.se_block(x_2d)
        x_2d = self.channel_attention(x_2d) * x_2d
        x_2d = self.residual_block(x_2d)
        
        x_2d = F.dropout(x_2d, p=0.3, training=self.training)
        
        # Second set of enhancements (no spatial attention)
        x_2d = self.se_block(x_2d)
        x_2d = self.channel_attention(x_2d) * x_2d
        x_2d = self.r2_block(x_2d)
        
        # Flatten back to 1D
        x = x_2d.squeeze(-1).squeeze(-1)  # [batch_size, feature_dim]
        x = self.fc(x)
        return x

class EnhancedHRNet(nn.Module):
    """Enhanced HRNet-W18 with custom attention and residual modules (CNN이므로 spatial attention 사용 가능)"""
    def __init__(self, num_classes=3):
        super().__init__()
        # HRNet backbone - global_pool과 classifier 제거하여 2D feature map 얻기
        self.hrnet = timm.create_model('hrnet_w18_small_v2', pretrained=False, num_classes=1000)
        
        # HRNet의 마지막 3개 레이어 제거 (global_pool, head_drop, classifier)
        self.hrnet.global_pool = nn.Identity()
        self.hrnet.head_drop = nn.Identity()
        self.hrnet.classifier = nn.Identity()
        
        # Get HRNet output dimension (2D feature map)
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_output = self.hrnet(test_input)
            hrnet_output_shape = test_output.shape
            hrnet_channels = test_output.shape[1]
        
        print(f"HRNet output shape: {hrnet_output_shape}")
        
        # Custom enhancement modules for HRNet features (spatial attention 사용 가능!)
        self.se_block = SEBlock(hrnet_channels)
        self.channel_attention = ChannelAttention(hrnet_channels)
        self.spatial_attention = EnhancedSpatialAttention()  # 복원!
        self.residual_block = BasicBlock(hrnet_channels, hrnet_channels)
        self.r2_block = R2Block(hrnet_channels, hrnet_channels)
        
        # Final layers
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hrnet_channels, num_classes)

    def forward(self, x):
        # HRNet feature extraction (2D feature map)
        x = self.hrnet(x)  # Shape: [batch_size, channels, H, W]
        
        # First set of enhancements (spatial attention 사용!)
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x  # 복원!
        x = self.residual_block(x)
        
        x = F.dropout(x, p=0.3, training=self.training)
        
        # Second set of enhancements (spatial attention 사용!)
        x = self.se_block(x)
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x  # 복원!
        x = self.r2_block(x)
        
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Enhanced Model Factory
class EnhancedModelFactory:
    """Factory class for creating enhanced models"""
    
    @staticmethod
    def get_enhanced_model(model_name, num_classes=3):
        """Get enhanced model by backbone name"""
        if model_name == 'resnet':
            return EnhancedResNet(num_classes)
        elif model_name == 'densenet':
            return EnhancedDenseNet(num_classes)
        elif model_name == 'mobilenet':
            return EnhancedMobileNet(num_classes)
        elif model_name == 'efficientnet':
            return EnhancedEfficientNet(num_classes)
        elif model_name == 'shufflenet':
            return EnhancedShuffleNet(num_classes)
        elif model_name == 'convnext':
            return EnhancedConvNeXt(num_classes)
        elif model_name == 'resnext':
            return EnhancedResNeXt(num_classes)
        elif model_name == 'vit':
            return EnhancedViT(num_classes)
        elif model_name == 'swin':
            return EnhancedSwinTransformer(num_classes)
        elif model_name == 'hrnet':
            return EnhancedHRNet(num_classes)
        else:
            raise ValueError(f"Unknown enhanced model: {model_name}")
    
    @staticmethod
    def get_available_models():
        """Get list of available enhanced model names"""
        return ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext', 'resnext', 'vit', 'swin', 'hrnet']