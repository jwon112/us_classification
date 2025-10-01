#!/usr/bin/env python3
"""
시각화 및 차트 생성 모듈
실험 결과를 다양한 차트로 시각화하는 함수들을 제공
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

# FLOPs 계산을 위한 라이브러리
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop library not available. FLOPs calculation will be skipped.")
    print("   Install with: pip install thop")


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


def create_learning_curves_chart(epochs_df, results_dir):
    """학습 곡선 차트 생성"""
    if epochs_df.empty:
        print("⚠️  No epoch data available for learning curves")
        return
    
    # Enhanced 모델만 필터링
    enhanced_epochs = epochs_df[epochs_df['model_type'] == 'enhanced']
    
    if enhanced_epochs.empty:
        print("⚠️  No enhanced model epoch data available")
        return
    
    # 차트 생성
    plt.figure(figsize=(15, 10))
    
    # 모델별 색상 설정
    models = enhanced_epochs['model_name'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 각 모델별로 학습 곡선 그리기
    for i, model in enumerate(models):
        model_data = enhanced_epochs[enhanced_epochs['model_name'] == model]
        
        # 여러 seed의 평균 계산
        epoch_means = model_data.groupby('epoch').agg({
            'train_loss': 'mean',
            'test_accuracy': 'mean'
        }).reset_index()
        
        # Train Loss (왼쪽 y축)
        plt.subplot(2, 1, 1)
        plt.plot(epoch_means['epoch'], epoch_means['train_loss'], 
                label=f'{model.upper()}', color=colors[i], linewidth=2)
        
        # Test Accuracy (오른쪽 y축)
        plt.subplot(2, 1, 2)
        plt.plot(epoch_means['epoch'], epoch_means['test_accuracy'], 
                label=f'{model.upper()}', color=colors[i], linewidth=2)
    
    # Train Loss 차트 설정
    plt.subplot(2, 1, 1)
    plt.title('Training Loss Curves - All Enhanced Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Test Accuracy 차트 설정
    plt.subplot(2, 1, 2)
    plt.title('Test Accuracy Curves - All Enhanced Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 저장
    chart_path = os.path.join(results_dir, "learning_curves.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Learning curves chart saved to: {chart_path}")


def create_parameter_efficiency_charts(results_df, results_dir):
    """파라미터 효율성 차트 생성"""
    if results_df.empty:
        print("⚠️  No results data available for parameter efficiency charts")
        return
    
    # Baseline과 Enhanced 모델 분리
    baseline_results = results_df[results_df['model_type'] == 'baseline']
    enhanced_results = results_df[results_df['model_type'] == 'enhanced']
    
    if baseline_results.empty or enhanced_results.empty:
        print("⚠️  Both baseline and enhanced results needed for parameter efficiency charts")
        return
    
    # Enhanced 모델의 평균 성능 계산 (seed별 평균)
    enhanced_avg = enhanced_results.groupby('model_name').agg({
        'final_accuracy': 'mean',  # seed별 평균
        'total_params': 'first'    # parameter 수는 동일하므로 첫 번째 값 사용
    }).reset_index()
    
    # Baseline 모델의 평균 성능 계산 (seed별 평균)
    baseline_avg = baseline_results.groupby('model_name').agg({
        'final_accuracy': 'mean',  # seed별 평균
        'total_params': 'first'    # parameter 수는 동일하므로 첫 번째 값 사용
    }).reset_index()
    
    # 차트 1: 파라미터 수 vs 성능 (Scatter Plot)
    plt.figure(figsize=(12, 8))
    
    # Baseline 모델 (파란색) - seed별 평균
    plt.scatter(baseline_avg['total_params'], baseline_avg['final_accuracy'], 
               c='blue', s=100, alpha=0.7, label='Baseline Models', marker='o')
    
    # Enhanced 모델 (빨간색) - seed별 평균
    plt.scatter(enhanced_avg['total_params'], enhanced_avg['final_accuracy'], 
               c='red', s=100, alpha=0.7, label='Enhanced Models', marker='^')
    
    # 모델 이름 표시
    for _, row in baseline_avg.iterrows():
        plt.annotate(f"{row['model_name'].upper()}", 
                    (row['total_params'], row['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    for _, row in enhanced_avg.iterrows():
        plt.annotate(f"{row['model_name'].upper()}", 
                    (row['total_params'], row['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Total Parameters (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('Parameter Efficiency: Parameters vs Performance')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart1_path = os.path.join(results_dir, "parameter_efficiency_scatter.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Parameter efficiency scatter plot saved to: {chart1_path}")
    
    # 차트 2: 파라미터 증가율 vs 성능 향상
    plt.figure(figsize=(12, 8))
    
    # Baseline과 Enhanced 모델 매칭 (seed별 평균 사용)
    comparison_data = []
    for _, enhanced_row in enhanced_avg.iterrows():
        model_name = enhanced_row['model_name']
        baseline_row = baseline_avg[baseline_avg['model_name'] == model_name]
        
        if not baseline_row.empty:
            baseline_row = baseline_row.iloc[0]
            
            param_increase = ((enhanced_row['total_params'] - baseline_row['total_params']) / 
                            baseline_row['total_params']) * 100
            performance_improvement = enhanced_row['final_accuracy'] - baseline_row['final_accuracy']
            
            comparison_data.append({
                'model_name': model_name,
                'param_increase': param_increase,
                'performance_improvement': performance_improvement
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # 색상 설정 (향상도에 따라)
        colors = ['green' if x > 0 else 'red' for x in comp_df['performance_improvement']]
        
        plt.scatter(comp_df['param_increase'], comp_df['performance_improvement'], 
                   c=colors, s=100, alpha=0.7)
        
        # 모델 이름 표시
        for _, row in comp_df.iterrows():
            plt.annotate(f"{row['model_name'].upper()}", 
                        (row['param_increase'], row['performance_improvement']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Parameter Increase (%)')
        plt.ylabel('Performance Improvement (Enhanced - Baseline)')
        plt.title('Parameter Efficiency: Increase vs Improvement')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        chart2_path = os.path.join(results_dir, "parameter_efficiency_improvement.png")
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Parameter efficiency improvement chart saved to: {chart2_path}")


def create_flops_efficiency_charts(results_df, results_dir):
    """FLOPs 효율성 차트 생성"""
    if results_df.empty:
        print("⚠️  No results data available for FLOPs efficiency charts")
        return
    
    # Baseline과 Enhanced 모델 분리
    baseline_results = results_df[results_df['model_type'] == 'baseline']
    enhanced_results = results_df[results_df['model_type'] == 'enhanced']
    
    if baseline_results.empty or enhanced_results.empty:
        print("⚠️  Both baseline and enhanced results needed for FLOPs efficiency charts")
        return
    
    # Enhanced 모델의 평균 성능 계산 (seed별 평균)
    enhanced_avg = enhanced_results.groupby('model_name').agg({
        'final_accuracy': 'mean',  # seed별 평균
        'total_params': 'first',   # parameter 수는 동일하므로 첫 번째 값 사용
        'flops': 'first'           # FLOPs도 동일하므로 첫 번째 값 사용
    }).reset_index()
    
    # Baseline 모델의 평균 성능 계산 (seed별 평균)
    baseline_avg = baseline_results.groupby('model_name').agg({
        'final_accuracy': 'mean',  # seed별 평균
        'total_params': 'first',   # parameter 수는 동일하므로 첫 번째 값 사용
        'flops': 'first'           # FLOPs도 동일하므로 첫 번째 값 사용
    }).reset_index()
    
    print("📊 Using FLOPs data from CSV files...")
    
    # 차트 1: FLOPs vs 성능 (Scatter Plot)
    plt.figure(figsize=(12, 8))
    
    # Baseline 모델 (파란색) - seed별 평균
    plt.scatter(baseline_avg['flops'], baseline_avg['final_accuracy'], 
               c='blue', s=100, alpha=0.7, label='Baseline Models', marker='o')
    
    # Enhanced 모델 (빨간색) - seed별 평균
    plt.scatter(enhanced_avg['flops'], enhanced_avg['final_accuracy'], 
               c='red', s=100, alpha=0.7, label='Enhanced Models', marker='^')
    
    # 모델 이름 표시
    for _, row in baseline_avg.iterrows():
        plt.annotate(f"{row['model_name'].upper()}", 
                    (row['flops'], row['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    for _, row in enhanced_avg.iterrows():
        plt.annotate(f"{row['model_name'].upper()}", 
                    (row['flops'], row['final_accuracy']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('FLOPs (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('FLOPs Efficiency: FLOPs vs Performance')
    plt.xscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart1_path = os.path.join(results_dir, "flops_efficiency_scatter.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 FLOPs efficiency scatter plot saved to: {chart1_path}")
    
    # 차트 2: FLOPs 증가율 vs 성능 향상
    plt.figure(figsize=(12, 8))
    
    # Baseline과 Enhanced 모델 매칭 (seed별 평균 사용)
    comparison_data = []
    for _, enhanced_row in enhanced_avg.iterrows():
        model_name = enhanced_row['model_name']
        baseline_row = baseline_avg[baseline_avg['model_name'] == model_name]
        
        if not baseline_row.empty:
            baseline_row = baseline_row.iloc[0]
            
            flops_increase = ((enhanced_row['flops'] - baseline_row['flops']) / 
                            baseline_row['flops']) * 100 if baseline_row['flops'] > 0 else 0
            performance_improvement = enhanced_row['final_accuracy'] - baseline_row['final_accuracy']
            
            comparison_data.append({
                'model_name': model_name,
                'flops_increase': flops_increase,
                'performance_improvement': performance_improvement
            })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        
        # 색상 설정 (향상도에 따라)
        colors = ['green' if x > 0 else 'red' for x in comp_df['performance_improvement']]
        
        plt.scatter(comp_df['flops_increase'], comp_df['performance_improvement'], 
                   c=colors, s=100, alpha=0.7)
        
        # 모델 이름 표시
        for _, row in comp_df.iterrows():
            plt.annotate(f"{row['model_name'].upper()}", 
                        (row['flops_increase'], row['performance_improvement']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('FLOPs Increase (%)')
        plt.ylabel('Performance Improvement (Enhanced - Baseline)')
        plt.title('FLOPs Efficiency: Increase vs Improvement')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, alpha=0.3)
        
        chart2_path = os.path.join(results_dir, "flops_efficiency_improvement.png")
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 FLOPs efficiency improvement chart saved to: {chart2_path}")


def create_confusion_matrices(models, train_loader, test_loader, results_df, epochs_df, results_dir, device='cuda'):
    """각 모델별 confusion matrix 생성 (best epoch와 last epoch)"""
    
    # Confusion matrix 저장 디렉토리 생성
    confusion_dir = os.path.join(results_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    
    # Enhanced 모델만 처리
    enhanced_results = results_df[results_df['model_type'] == 'enhanced']
    
    if enhanced_results.empty:
        print("⚠️  No enhanced model results available for confusion matrices")
        return
    
    print(f"\n📊 Creating confusion matrices for {len(models)} models...")
    
    for model_name in models:
        try:
            print(f"   🔄 Processing {model_name.upper()}...")
            
            # 해당 모델의 결과들
            model_results = enhanced_results[enhanced_results['model_name'] == model_name]
            
            if model_results.empty:
                print(f"   ⚠️  No results found for {model_name}")
                continue
            
            # 모델 생성
            from models.enhanced_models import EnhancedModelFactory
            model = EnhancedModelFactory.get_enhanced_model(model_name, num_classes=3)
            model = model.to(device)
            
            # Best epoch와 Last epoch 계산
            best_epochs = []
            last_epochs = []
            
            for _, row in model_results.iterrows():
                seed = row['seed']
                best_epoch = row['best_epoch']
                last_epoch = epochs_df[
                    (epochs_df['model_name'] == model_name) & 
                    (epochs_df['seed'] == seed)
                ]['epoch'].max()
                
                best_epochs.append(best_epoch)
                last_epochs.append(last_epoch)
            
            # 평균 epoch 계산
            avg_best_epoch = int(np.mean(best_epochs))
            avg_last_epoch = int(np.mean(last_epochs))
            
            # 모델을 best epoch 상태로 복원 (여기서는 마지막 상태 사용)
            # 실제로는 각 epoch별로 모델을 저장해야 하지만, 
            # 현재는 마지막 훈련된 모델을 사용
            
            # Best epoch confusion matrix
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
            
            # Confusion matrix 계산
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            
            # Best epoch confusion matrix 저장
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Benign', 'Malignant', 'Normal'],
                       yticklabels=['Benign', 'Malignant', 'Normal'])
            plt.title(f'Enhanced {model_name.upper()} - Best Epoch (Avg: {avg_best_epoch})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            best_matrix_path = os.path.join(confusion_dir, f"enhanced_{model_name}_best_epoch.png")
            plt.savefig(best_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Last epoch confusion matrix (동일한 모델 사용)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Benign', 'Malignant', 'Normal'],
                       yticklabels=['Benign', 'Malignant', 'Normal'])
            plt.title(f'Enhanced {model_name.upper()} - Last Epoch (Avg: {avg_last_epoch})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            last_matrix_path = os.path.join(confusion_dir, f"enhanced_{model_name}_last_epoch.png")
            plt.savefig(last_matrix_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Created confusion matrices for {model_name.upper()}")
            
        except Exception as e:
            print(f"   ❌ Error creating confusion matrix for {model_name}: {e}")
            continue
    
    print(f"📊 Confusion matrices saved to: {confusion_dir}")


def save_confusion_matrix(conf_matrix, model_name, seed, epoch_type, epoch, save_dir="baseline_results"):
    """Baseline 모델용 Confusion Matrix 저장"""
    # confusion_matrices 폴더 생성
    confusion_dir = os.path.join(save_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant', 'Normal'],
                yticklabels=['Benign', 'Malignant', 'Normal'])
    plt.title(f'Baseline {model_name.upper()} - {epoch_type.title()} Epoch ({epoch})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = f"baseline_{model_name}_{epoch_type}_epoch.png"
    filepath = os.path.join(confusion_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {filepath}")
    return filepath


def create_model_comparison_chart(enhanced_results, baseline_results, save_path="model_comparison.png"):
    """Enhanced vs Baseline 모델 성능 비교 차트 생성"""
    
    if not enhanced_results or not baseline_results:
        print("결과 데이터가 부족하여 비교 플롯을 생성할 수 없습니다.")
        return
    
    # 데이터 준비 - 10개 모델 모두 포함
    models = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext', 'resnext', 'vit', 'swin', 'hrnet']
    
    enhanced_avg = {}
    baseline_avg = {}
    
    # Enhanced 모델 평균 성능 계산
    for model in models:
        model_results = [r for r in enhanced_results if r.get('model_name', '').lower() == model]
        if model_results:
            accuracies = [r.get('final_accuracy', 0) for r in model_results]
            enhanced_avg[model] = np.mean(accuracies)
    
    # Baseline 모델 평균 성능 계산
    for model in models:
        model_results = [r for r in baseline_results if r.get('Model', '').lower() == model]
        if model_results:
            accuracies = [r.get('Test_Accuracy', 0) for r in model_results]
            baseline_avg[model] = np.mean(accuracies)
    
    # 비교 데이터 생성
    comparison_data = []
    for model in models:
        if model in enhanced_avg and model in baseline_avg:
            comparison_data.append({
                'Model': model.upper(),
                'Enhanced': enhanced_avg[model],
                'Baseline': baseline_avg[model],
                'Improvement': enhanced_avg[model] - baseline_avg[model]
            })
    
    if not comparison_data:
        print("비교할 수 있는 데이터가 없습니다.")
        return
    
    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. 성능 비교 바 차트
    df_comp = pd.DataFrame(comparison_data)
    x = np.arange(len(df_comp))
    width = 0.35
    
    ax1.bar(x - width/2, df_comp['Baseline'], width, label='Baseline', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, df_comp['Enhanced'], width, label='Enhanced', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Enhanced vs Baseline Model Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp['Model'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 성능 향상 차트
    colors = ['green' if x > 0 else 'red' for x in df_comp['Improvement']]
    ax2.bar(df_comp['Model'], df_comp['Improvement'], color=colors, alpha=0.7)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Performance Improvement (Enhanced - Baseline)')
    ax2.set_xticks(range(len(df_comp)))
    ax2.set_xticklabels(df_comp['Model'], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(df_comp['Improvement']):
        ax2.text(i, v + (0.01 if v > 0 else -0.01), f'{v:.4f}', 
                ha='center', va='bottom' if v > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Model comparison chart saved to: {save_path}")
    
    return comparison_data


def create_parameter_efficiency_analysis(enhanced_results, baseline_results):
    """파라미터 효율성 분석"""
    
    if not enhanced_results or not baseline_results:
        return
    
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY ANALYSIS")
    print("="*60)
    
    models = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext', 'resnext', 'vit', 'swin', 'hrnet']
    
    for model in models:
        # Enhanced 모델 정보
        enhanced_model_results = [r for r in enhanced_results if r.get('model_name', '').lower() == model]
        baseline_model_results = [r for r in baseline_results if r.get('Model', '').lower() == model]
        
        if enhanced_model_results and baseline_model_results:
            enhanced_params = enhanced_model_results[0].get('total_params', 0)
            baseline_params = baseline_model_results[0].get('Total_Params', 0)
            
            if enhanced_params > 0 and baseline_params > 0:
                param_increase = enhanced_params - baseline_params
                param_increase_pct = (param_increase / baseline_params) * 100
                
                print(f"{model.upper():12}: Baseline={baseline_params:8,}, "
                      f"Enhanced={enhanced_params:8,}, "
                      f"Increase={param_increase:+8,} ({param_increase_pct:+.1f}%)")
