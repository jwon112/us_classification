#!/usr/bin/env python3
"""
통합 실험 결과 분석 및 시각화
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_results(results_dir):
    """실험 결과 로드"""
    csv_path = os.path.join(results_dir, "integrated_experiment_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} results from {csv_path}")
    return df

def create_performance_comparison(df, save_dir):
    """성능 비교 차트 생성"""
    
    # Baseline vs Enhanced 비교
    plt.figure(figsize=(15, 10))
    
    # 1. Accuracy 비교
    plt.subplot(2, 2, 1)
    baseline_acc = df[df['model_type'] == 'baseline'].groupby('model_name')['final_accuracy'].mean()
    enhanced_acc = df[df['model_type'] == 'enhanced'].groupby('model_name')['final_accuracy'].mean()
    
    x = np.arange(len(baseline_acc))
    width = 0.35
    
    plt.bar(x - width/2, baseline_acc.values, width, label='Baseline', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, enhanced_acc.values, width, label='Enhanced', alpha=0.8, color='orange')
    
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Baseline vs Enhanced Models - Accuracy Comparison')
    plt.xticks(x, [name.upper() for name in baseline_acc.index], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. F1 Score 비교
    plt.subplot(2, 2, 2)
    baseline_f1 = df[df['model_type'] == 'baseline'].groupby('model_name')['final_f1'].mean()
    enhanced_f1 = df[df['model_type'] == 'enhanced'].groupby('model_name')['final_f1'].mean()
    
    plt.bar(x - width/2, baseline_f1.values, width, label='Baseline', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, enhanced_f1.values, width, label='Enhanced', alpha=0.8, color='orange')
    
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.title('Baseline vs Enhanced Models - F1 Score Comparison')
    plt.xticks(x, [name.upper() for name in baseline_f1.index], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Parameter Count 비교
    plt.subplot(2, 2, 3)
    baseline_params = df[df['model_type'] == 'baseline'].groupby('model_name')['total_params'].first()
    enhanced_params = df[df['model_type'] == 'enhanced'].groupby('model_name')['total_params'].first()
    
    plt.bar(x - width/2, baseline_params.values, width, label='Baseline', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, enhanced_params.values, width, label='Enhanced', alpha=0.8, color='orange')
    
    plt.xlabel('Model')
    plt.ylabel('Parameter Count')
    plt.title('Baseline vs Enhanced Models - Parameter Count Comparison')
    plt.xticks(x, [name.upper() for name in baseline_params.index], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 4. Accuracy vs Parameters (Scatter plot)
    plt.subplot(2, 2, 4)
    for model_type in ['baseline', 'enhanced']:
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['total_params'], subset['final_accuracy'], 
                   label=model_type.capitalize(), alpha=0.7, s=50)
    
    plt.xlabel('Parameter Count')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Parameter Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Performance comparison chart saved")

def create_seed_analysis(df, save_dir):
    """시드별 분석 차트"""
    
    plt.figure(figsize=(15, 10))
    
    # 1. 시드별 평균 성능
    plt.subplot(2, 2, 1)
    seed_performance = df.groupby(['seed', 'model_type'])['final_accuracy'].mean().unstack()
    seed_performance.plot(kind='bar', ax=plt.gca())
    plt.title('Average Performance by Seed and Model Type')
    plt.xlabel('Seed')
    plt.ylabel('Average Accuracy')
    plt.legend(title='Model Type')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 2. 시드별 모델 성능 분포
    plt.subplot(2, 2, 2)
    for model_type in ['baseline', 'enhanced']:
        subset = df[df['model_type'] == model_type]
        plt.hist(subset['final_accuracy'], alpha=0.7, label=model_type.capitalize(), bins=15)
    
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.title('Accuracy Distribution by Model Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 시드별 최고 성능 모델
    plt.subplot(2, 2, 3)
    best_per_seed = df.loc[df.groupby('seed')['final_accuracy'].idxmax()]
    best_per_seed.plot(x='seed', y='final_accuracy', kind='bar', ax=plt.gca())
    plt.title('Best Model Performance by Seed')
    plt.xlabel('Seed')
    plt.ylabel('Best Accuracy')
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    
    # 4. 시드별 성능 변동성
    plt.subplot(2, 2, 4)
    seed_variability = df.groupby('seed')['final_accuracy'].std()
    plt.bar(seed_variability.index, seed_variability.values, alpha=0.8, color='lightcoral')
    plt.title('Performance Variability by Seed (Standard Deviation)')
    plt.xlabel('Seed')
    plt.ylabel('Standard Deviation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Seed analysis chart saved")

def create_model_ranking(df, save_dir):
    """모델 순위 차트"""
    
    # 전체 평균 성능으로 순위 결정
    model_ranking = df.groupby(['model_type', 'model_name'])['final_accuracy'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    # 색상 매핑
    colors = ['skyblue' if 'baseline' in idx else 'orange' for idx in model_ranking.index]
    
    bars = plt.bar(range(len(model_ranking)), model_ranking.values, color=colors, alpha=0.8)
    
    # 라벨 설정
    labels = [f"{model_type.upper()}_{name.upper()}" for (model_type, name) in model_ranking.index]
    plt.xticks(range(len(model_ranking)), labels, rotation=45, ha='right')
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, model_ranking.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.title('Model Performance Ranking (All Models)')
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', alpha=0.8, label='Baseline'),
                      Patch(facecolor='orange', alpha=0.8, label='Enhanced')]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_ranking.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Model ranking chart saved")
    
    # 순위 출력
    print("\n🏆 MODEL PERFORMANCE RANKING:")
    print("="*50)
    for i, ((model_type, model_name), accuracy) in enumerate(model_ranking.items(), 1):
        print(f"{i:2d}. {model_type.upper():10} {model_name.upper():12} | Accuracy: {accuracy:.4f}")

def create_detailed_statistics(df, save_dir):
    """상세 통계 생성"""
    
    # 요약 통계
    summary_stats = df.groupby(['model_type', 'model_name']).agg({
        'final_accuracy': ['count', 'mean', 'std', 'min', 'max'],
        'final_f1': ['mean', 'std'],
        'final_precision': ['mean', 'std'],
        'final_recall': ['mean', 'std'],
        'total_params': 'first',
        'best_accuracy': ['mean', 'std']
    }).round(4)
    
    # CSV로 저장
    summary_path = os.path.join(save_dir, "detailed_statistics.csv")
    summary_stats.to_csv(summary_path)
    
    # LaTeX 테이블로도 저장
    latex_path = os.path.join(save_dir, "detailed_statistics.tex")
    with open(latex_path, 'w') as f:
        f.write(summary_stats.to_latex())
    
    print(f"📈 Detailed statistics saved to CSV and LaTeX")
    
    # 통계 출력
    print("\n📊 DETAILED STATISTICS:")
    print("="*80)
    print(summary_stats)
    
    return summary_stats

def main():
    parser = argparse.ArgumentParser(description='Analyze Integrated Experiment Results')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Path to results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for analysis (default: results_dir)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Starting Integrated Results Analysis")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    
    try:
        # 결과 로드
        df = load_results(args.results_dir)
        
        # 기본 정보 출력
        print(f"\n📊 Dataset Overview:")
        print(f"   Total experiments: {len(df)}")
        print(f"   Seeds: {sorted(df['seed'].unique())}")
        print(f"   Model types: {df['model_type'].unique()}")
        print(f"   Models: {sorted(df['model_name'].unique())}")
        
        # 차트 생성
        create_performance_comparison(df, args.output_dir)
        create_seed_analysis(df, args.output_dir)
        create_model_ranking(df, args.output_dir)
        
        # 상세 통계
        summary_stats = create_detailed_statistics(df, args.output_dir)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"📂 Results saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
