import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class BaselineResNet(nn.Module):
    """순수 ResNet50 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=None)
        # classifier만 교체
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineDenseNet(nn.Module):
    """순수 DenseNet121 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        densenet = models.densenet121(weights=None)
        # classifier만 교체
        self.features = densenet.features
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineMobileNet(nn.Module):
    """순수 MobileNetV2 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        # classifier만 교체
        self.features = mobilenet.features
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineEfficientNet(nn.Module):
    """순수 EfficientNet-B0 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        efficientnet = models.efficientnet_b0(weights=None)
        # classifier만 교체
        self.features = efficientnet.features
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineShuffleNet(nn.Module):
    """순수 ShuffleNetV2 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        shufflenet = models.shufflenet_v2_x1_0(weights=None)
        # classifier만 교체
        self.conv1 = shufflenet.conv1
        self.stage2 = shufflenet.stage2
        self.stage3 = shufflenet.stage3
        self.stage4 = shufflenet.stage4
        self.conv5 = shufflenet.conv5
        self.fc = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineConvNeXt(nn.Module):
    """순수 ConvNeXt-Tiny 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        convnext = models.convnext_tiny(weights=None)
        # classifier만 교체
        self.features = convnext.features
        self.fc = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineResNeXt(nn.Module):
    """순수 ResNeXt50 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        resnext = models.resnext50_32x4d(weights=None)
        # classifier만 교체
        self.features = nn.Sequential(*list(resnext.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BaselineViT(nn.Module):
    """순수 ViT-B/16 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.vit(x)

class BaselineSwinTransformer(nn.Module):
    """순수 Swin-T 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.swin(x)

class BaselineHRNet(nn.Module):
    """순수 HRNet-W18 백본 (custom layer 없음)"""
    def __init__(self, num_classes):
        super().__init__()
        self.hrnet = timm.create_model('hrnet_w18_small_v2', pretrained=False, num_classes=num_classes)
    
    def forward(self, x):
        return self.hrnet(x)

# 모델 선택 함수
def get_baseline_model(model_name, num_classes=3):
    """순수 백본 모델 반환"""
    if model_name == 'resnet':
        return BaselineResNet(num_classes)
    elif model_name == 'densenet':
        return BaselineDenseNet(num_classes)
    elif model_name == 'mobilenet':
        return BaselineMobileNet(num_classes)
    elif model_name == 'efficientnet':
        return BaselineEfficientNet(num_classes)
    elif model_name == 'shufflenet':
        return BaselineShuffleNet(num_classes)
    elif model_name == 'convnext':
        return BaselineConvNeXt(num_classes)
    elif model_name == 'resnext':
        return BaselineResNeXt(num_classes)
    elif model_name == 'vit':
        return BaselineViT(num_classes)
    elif model_name == 'swin':
        return BaselineSwinTransformer(num_classes)
    elif model_name == 'hrnet':
        return BaselineHRNet(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# 모델 정보 출력 함수
def print_model_info(model, model_name):
    """모델의 기본 정보 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== {model_name.upper()} Baseline Model ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 테스트 forward pass
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224)
        test_output = model(test_input)
        print(f"Output shape: {test_output.shape}")
        print(f"Output classes: {test_output.shape[1]}")
    
    print("=" * 50)
    return total_params, trainable_params

if __name__ == "__main__":
    # 테스트
    print("Baseline 모델 테스트")
    print("=" * 50)
    
    models_to_test = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext', 'resnext', 'vit', 'swin', 'hrnet']
    
    for model_name in models_to_test:
        try:
            model = get_baseline_model(model_name, num_classes=3)
            total_params, trainable_params = print_model_info(model, model_name)
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            print("=" * 50)