#!/usr/bin/env python3
"""
통합 실험 시스템
Baseline 모델 결과를 로드하고 Enhanced 모델들만 훈련하여 비교
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import wandb
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 import
from models.backbones import get_baseline_model
from models.enhanced_models import EnhancedModelFactory

def set_seed(seed):
    """랜덤 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(data_path, batch_size=32):
    """데이터 로더 생성"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(os.path.join(data_path, 'train'), transform=transform)
    test_dataset = ImageFolder(os.path.join(data_path, 'test'), transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cuda'):
    """모델 훈련"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct / total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Test Acc: {test_accuracy:.2f}%')
    
    return train_losses, test_accuracies

def evaluate_model(model, test_loader, device='cuda'):
    """모델 평가"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 메트릭 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

def load_baseline_results():
    """저장된 baseline 결과 로드"""
    baseline_csv_path = "baseline_results/baseline_results_summary.csv"
    
    if not os.path.exists(baseline_csv_path):
        print(f"⚠️  Warning: {baseline_csv_path} not found!")
        print("   Please run train_baseline.py first to generate baseline results.")
        return None
    
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        print(f"✅ Loaded baseline results from: {baseline_csv_path}")
        return baseline_df
    except Exception as e:
        print(f"❌ Error loading baseline results: {e}")
        return None

def run_integrated_experiment(data_path, epochs=10, batch_size=32, seeds=[24], models=None):
    """통합 실험 실행"""
    
    # 실험 결과 저장 디렉토리 (experiments/ 폴더 안에)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/integrated_experiment_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 사용 가능한 모델들 (새로운 모델들 포함)
    if models is None:
        enhanced_models = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext', 'resnext', 'vit', 'swin', 'hrnet']
    else:
        enhanced_models = models
    
    # Baseline 결과 로드
    print("\n📊 Loading Baseline Results...")
    baseline_df = load_baseline_results()
    
    if baseline_df is None:
        print("❌ Cannot proceed without baseline results. Exiting.")
        return None, None
    
    # 결과 저장용
    all_results = []
    
    # Baseline 결과를 all_results에 추가
    print("\n📋 Adding Baseline Results to Comparison...")
    for _, row in baseline_df.iterrows():
        baseline_result = {
            'seed': 'baseline',  # baseline은 seed가 없음
            'model_type': 'baseline',
            'model_name': row['Model'],
            'total_params': row['Total_Params'],
            'final_accuracy': row['Test_Accuracy'],
            'final_f1': row['F1_Score'],
            'final_precision': row['Precision'],
            'final_recall': row['Recall'],
            'best_epoch': row['Best_Epoch'],
            'best_accuracy': row['Test_Accuracy']  # baseline에는 best_accuracy 컬럼이 없어서 Test_Accuracy 사용
        }
        all_results.append(baseline_result)
        print(f"   ✅ Added {row['Model'].upper()} baseline results")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Enhanced 모델들만 훈련
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"🔄 Training Enhanced Models with seed: {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        # 데이터 로더 생성
        train_loader, test_loader = get_data_loaders(data_path, batch_size)
        
        # Enhanced 모델들 훈련
        print("\n🚀 Training Enhanced Models...")
        for model_name in enhanced_models:
            try:
                print(f"\n🔄 Training Enhanced {model_name.upper()}...")
                
                # 모델 생성
                model = EnhancedModelFactory.get_enhanced_model(model_name, num_classes=3)
                
                # 파라미터 수 계산
                total_params = sum(p.numel() for p in model.parameters())
                
                # 훈련
                train_losses, test_accuracies = train_model(
                    model, train_loader, test_loader, epochs, device=device
                )
                
                # 평가
                metrics = evaluate_model(model, test_loader, device)
                
                # 결과 저장
                result = {
                    'seed': seed,
                    'model_type': 'enhanced',
                    'model_name': model_name,
                    'total_params': total_params,
                    'final_accuracy': metrics['accuracy'],
                    'final_f1': metrics['f1'],
                    'final_precision': metrics['precision'],
                    'final_recall': metrics['recall'],
                    'best_epoch': np.argmax(test_accuracies) + 1,
                    'best_accuracy': max(test_accuracies)
                }
                all_results.append(result)
                
                print(f"✅ Enhanced {model_name.upper()} completed")
                print(f"   Final Accuracy: {metrics['accuracy']:.4f}")
                print(f"   Best Accuracy: {max(test_accuracies):.2f}% at epoch {np.argmax(test_accuracies) + 1}")
                
            except Exception as e:
                print(f"❌ Error with Enhanced {model_name}: {e}")
                continue
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(all_results)
    
    # CSV로 저장
    csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n📊 Results saved to: {csv_path}")
    
    # 요약 통계 생성
    summary_stats = results_df.groupby(['model_type', 'model_name']).agg({
        'final_accuracy': ['mean', 'std'],
        'final_f1': ['mean', 'std'],
        'total_params': 'first'
    }).round(4)
    
    summary_path = os.path.join(results_dir, "summary_statistics.csv")
    summary_stats.to_csv(summary_path)
    print(f"📈 Summary statistics saved to: {summary_path}")
    
    # 결과 출력
    print("\n" + "="*80)
    print("INTEGRATED EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    print("\n--- Baseline Models (Loaded from saved results) ---")
    baseline_results = results_df[results_df['model_type'] == 'baseline']
    for _, row in baseline_results.iterrows():
        print(f"{row['model_name'].upper():12} | Acc: {row['final_accuracy']:.4f} | Params: {row['total_params']:,}")
    
    print("\n--- Enhanced Models (Trained in this experiment) ---")
    enhanced_results = results_df[results_df['model_type'] == 'enhanced']
    enhanced_summary = enhanced_results.groupby('model_name').agg({
        'final_accuracy': ['mean', 'std'],
        'total_params': 'first'
    }).round(4)
    
    for model_name in enhanced_summary.index:
        acc_mean = enhanced_summary.loc[model_name, ('final_accuracy', 'mean')]
        acc_std = enhanced_summary.loc[model_name, ('final_accuracy', 'std')]
        params = enhanced_summary.loc[model_name, ('total_params', 'first')]
        print(f"{model_name.upper():12} | Acc: {acc_mean:.4f} ± {acc_std:.4f} | Params: {params:,}")
    
    return results_dir, results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Integrated Experiment System')
    parser.add_argument('--data_path', type=str, default='DATA/GDPHSYSUCC/GDPHnSYSUCC_all', 
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--seeds', nargs='+', type=int, default=[24], 
                       help='Random seeds for experiments')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                       help='Specific models to train (default: all models)')
    
    args = parser.parse_args()
    
    print("🚀 Starting Integrated Experiment System")
    print(f"📁 Data path: {args.data_path}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎲 Seeds: {args.seeds}")
    print(f"🤖 Models: {args.models if args.models else 'All models'}")
    print(f"📂 Results will be saved in: experiments/ folder")
    
    try:
        results_dir, results_df = run_integrated_experiment(
            args.data_path, args.epochs, args.batch_size, args.seeds, args.models
        )
        
        if results_dir and results_df is not None:
            print(f"\n🎉 Integrated experiment completed successfully!")
            print(f"📂 Results saved in: {results_dir}")
        else:
            print(f"\n❌ Experiment failed or baseline results not found.")
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()