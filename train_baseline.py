import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import argparse
from models.backbones import get_baseline_model
from visualization import save_confusion_matrix

# FLOPs 계산을 위한 라이브러리
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. FLOPs calculation will be skipped.")
    print("   Install with: pip install thop")

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(seed):
    """랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """모델의 파라미터 개수를 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def calculate_model_size_mb(model):
    """모델 크기를 MB 단위로 계산"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """모델의 FLOPs 계산"""
    if not THOP_AVAILABLE:
        return 0
    
    try:
        # 모델의 디바이스 확인
        device = next(model.parameters()).device
        
        # 더미 입력을 모델과 같은 디바이스에 생성
        dummy_input = torch.randn(input_size).to(device)
        
        # FLOPs 계산
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops
    except Exception as e:
        print(f"⚠️  Error calculating FLOPs: {e}")
        return 0

def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs, model_name, seed):
    """모델 훈련 및 평가"""
    print(f"\nTraining {model_name.upper()} Baseline Model with Seed {seed}")
    print("=" * 60)
    
    # 결과 저장용 리스트
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    best_epoch = 0
    best_confusion_matrix = None
    last_confusion_matrix = None
    
    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders['train'].dataset)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())
        
        # 테스트 단계
        model.eval()
        test_corrects = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = test_corrects.double() / len(dataloaders['test'].dataset)
        test_accuracies.append(test_acc.item())
        
        # 메트릭 계산
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
        print(f"Test Acc: {test_acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print("-" * 40)
        
        # 최고 성능 모델 저장
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            best_confusion_matrix = conf_matrix
            
            # 모델 저장
            model_save_path = os.path.join("baseline_results", f"baseline_{model_name}_seed_{seed}_best.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved: {model_save_path}")
        
        # 마지막 epoch의 confusion matrix 저장
        if epoch == num_epochs - 1:
            last_confusion_matrix = conf_matrix
    
    return test_accuracies[-1], best_epoch, best_confusion_matrix, last_confusion_matrix, f1, precision, recall

def main():
    parser = argparse.ArgumentParser(description='Train Baseline CNN Models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seeds', nargs='+', type=int, default=[24], help='List of seeds')
    parser.add_argument('--models', nargs='+', type=str, 
                       default=['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext','resnext', 'vit', 'swin', 'hrnet'],
                       choices=['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext','resnext', 'vit', 'swin', 'hrnet'],
                       help='Models to train')
    
    args = parser.parse_args()
    
    print("Baseline CNN Models Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Seeds: {args.seeds}")
    print(f"Models: {args.models}")
    print("=" * 60)
    
    # 데이터 변환 설정
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 결과 저장 폴더 생성
    os.makedirs("baseline_results", exist_ok=True)
    print("Created/verified baseline_results directory")
    
    # 데이터 경로 설정
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(BASE_DIR, "DATA", "BUSI", "BUSI", "train")
    test_path = os.path.join(BASE_DIR, "DATA", "BUSI", "BUSI", "test")
    
    # 경로 확인
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train path does not exist: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test path does not exist: {test_path}")
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    dataloaders = {'train': train_dataloader, 'test': test_dataloader}
    
    # 결과 저장용 리스트
    all_results = []
    best_accuracy = 0.0
    best_seed = None
    best_model_name = None
    
    # 각 모델과 시드에 대해 훈련
    for model_name in args.models:
        for seed in args.seeds:
            print(f"\n{'='*20} Training {model_name.upper()} Baseline Model {'='*20}")
            
            # 시드 설정
            set_seed(seed)
            
            # 모델 생성
            model = get_baseline_model(model_name, num_classes=3).to(device)
            
            # 모델 정보 출력
            print(f"\n=== {model_name.upper()} Baseline Model Information ===")
            total_params, trainable_params = count_parameters(model)
            model_size_mb = calculate_model_size_mb(model)
            flops = calculate_flops(model)
            print(f"Model size: {model_size_mb:.2f} MB")
            print(f"FLOPs: {flops:,}")
            print("=" * 50)
            
            # 손실 함수 및 옵티마이저
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            # 모델 훈련 및 평가
            test_accuracy, best_epoch, best_confusion_matrix, last_confusion_matrix, best_f1, best_precision, best_recall = train_and_evaluate_model(
                model, dataloaders, criterion, optimizer, args.epochs, model_name, seed
            )
            
            # Confusion Matrix 저장
            if best_confusion_matrix is not None:
                save_confusion_matrix(best_confusion_matrix, model_name, seed, "best", best_epoch)
            if last_confusion_matrix is not None:
                save_confusion_matrix(last_confusion_matrix, model_name, seed, "last", args.epochs)
            
            # 결과 저장
            all_results.append([model_name, seed, total_params, trainable_params, model_size_mb, flops,
                              test_accuracy, best_epoch, best_f1, best_precision, best_recall])
            
            # 최고 성능 모델 업데이트
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_seed = seed
                best_model_name = model_name
    
    # 전체 결과 요약
    print("\n" + "="*80)
    print("BASELINE MODELS TRAINING RESULTS SUMMARY")
    print("="*80)
    
    # 결과를 DataFrame으로 변환
    columns = ['Model', 'Seed', 'Total_Params', 'Trainable_Params', 'Model_Size_MB', 'FLOPs',
               'Test_Accuracy', 'Best_Epoch', 'F1_Score', 'Precision', 'Recall']
    results_df = pd.DataFrame(all_results, columns=columns)
    
    # CSV로 저장
    csv_save_path = os.path.join("baseline_results", "baseline_results_summary.csv")
    results_df.to_csv(csv_save_path, index=False)
    print(f"Results saved to: {csv_save_path}")
    
    # 결과 출력
    for _, row in results_df.iterrows():
        print(f"Model: {row['Model'].upper():12} | Seed: {row['Seed']:3} | "
              f"Params: {row['Total_Params']:8,} | Size: {row['Model_Size_MB']:6.2f}MB | "
              f"Acc: {row['Test_Accuracy']:6.4f} | F1: {row['F1_Score']:6.4f}")
    
    # 최고 성능 모델 정보
    if best_model_name:
        print(f"\n🏆 BEST BASELINE MODEL: {best_model_name.upper()} with Seed {best_seed}")
        print(f"   Test Accuracy: {best_accuracy:.4f}")
    
    # 모델별 평균 성능
    print("\n" + "="*50)
    print("AVERAGE PERFORMANCE BY MODEL")
    print("="*50)
    
    model_avg = results_df.groupby('Model')[['Test_Accuracy', 'F1_Score', 'Precision', 'Recall']].mean()
    for model, metrics in model_avg.iterrows():
        print(f"{model.upper():12}: Acc={metrics['Test_Accuracy']:6.4f}, "
              f"F1={metrics['F1_Score']:6.4f}, P={metrics['Precision']:6.4f}, R={metrics['Recall']:6.4f}")

if __name__ == "__main__":
    main()