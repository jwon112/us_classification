import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_results():
    """결과 파일들을 로드하여 비교 데이터 생성"""
    
    # Enhanced 모델 결과 로드 (기존 experiment_results 폴더에서)
    enhanced_results = []
    experiment_dirs = [d for d in os.listdir('.') if d.startswith('experiment_results_')]
    
    if experiment_dirs:
        # 가장 최근 실험 결과 사용
        latest_dir = max(experiment_dirs)
        print(f"Loading enhanced model results from: {latest_dir}")
        
        # 우선적으로 overall_model_comparison_results.csv 찾기
        overall_csv_path = os.path.join(latest_dir, "overall_model_comparison_results.csv")
        if os.path.exists(overall_csv_path):
            enhanced_df = pd.read_csv(overall_csv_path)
            enhanced_results = enhanced_df.to_dict('records')
            print(f"   Loaded: overall_model_comparison_results.csv ({len(enhanced_results)} entries)")
        else:
            # overall 파일이 없으면 다른 CSV 파일들 중에서 찾기
            csv_files = [f for f in os.listdir(latest_dir) if f.endswith('.csv')]
            if csv_files:
                print(f"   Warning: overall_model_comparison_results.csv not found")
                print(f"   Available CSV files: {csv_files}")
                # 첫 번째 CSV 파일 로드 (기존 방식)
                enhanced_df = pd.read_csv(os.path.join(latest_dir, csv_files[0]))
                enhanced_results = enhanced_df.to_dict('records')
                print(f"   Loaded: {csv_files[0]} ({len(enhanced_results)} entries)")
    
    # Baseline 모델 결과 로드
    baseline_results = []
    baseline_csv_path = os.path.join("baseline_results", "baseline_results_summary.csv")
    if os.path.exists(baseline_csv_path):
        baseline_df = pd.read_csv(baseline_csv_path)
        baseline_results = baseline_df.to_dict('records')
        print(f"Loaded baseline results from: {baseline_csv_path}")
        print(f"   Found {len(baseline_results)} entries")
    else:
        print(f"Baseline results not found at: {baseline_csv_path}")
    
    return enhanced_results, baseline_results

def create_comparison_plot(enhanced_results, baseline_results):
    """Enhanced vs Baseline 모델 성능 비교 플롯 생성"""
    
    if not enhanced_results or not baseline_results:
        print("결과 데이터가 부족하여 비교 플롯을 생성할 수 없습니다.")
        return
    
    # 데이터 준비
    models = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext']
    
    enhanced_avg = {}
    baseline_avg = {}
    
    # Enhanced 모델 평균 성능 계산
    for model in models:
        model_results = [r for r in enhanced_results if r.get('Model', '').lower() == model]
        if model_results:
            accuracies = [r.get('Test_Accuracy', 0) for r in model_results]
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
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
    ax2.set_xticklabels(df_comp['Model'], rotation=45)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 값 표시
    for i, v in enumerate(df_comp['Improvement']):
        ax2.text(i, v + (0.01 if v > 0 else -0.01), f'{v:.4f}', 
                ha='center', va='bottom' if v > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('model_comparison_enhanced_vs_baseline.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 결과 출력
    print("\n" + "="*80)
    print("ENHANCED vs BASELINE MODEL COMPARISON")
    print("="*80)
    
    for row in comparison_data:
        print(f"{row['Model']:12}: Baseline={row['Baseline']:6.4f}, "
              f"Enhanced={row['Enhanced']:6.4f}, Improvement={row['Improvement']:+6.4f}")
    
    # 전체 평균 향상도
    avg_improvement = np.mean([r['Improvement'] for r in comparison_data])
    print(f"\n📊 Average Improvement: {avg_improvement:+.4f}")
    
    if avg_improvement > 0:
        print("✅ Enhanced models show overall improvement over baseline models!")
    else:
        print("⚠️  Enhanced models show no overall improvement over baseline models.")

def analyze_parameter_efficiency(enhanced_results, baseline_results):
    """파라미터 효율성 분석"""
    
    if not enhanced_results or not baseline_results:
        return
    
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY ANALYSIS")
    print("="*60)
    
    models = ['resnet', 'densenet', 'mobilenet', 'efficientnet', 'shufflenet', 'convnext']
    
    for model in models:
        # Enhanced 모델 정보
        enhanced_model_results = [r for r in enhanced_results if r.get('Model', '').lower() == model]
        baseline_model_results = [r for r in baseline_results if r.get('Model', '').lower() == model]
        
        if enhanced_model_results and baseline_model_results:
            enhanced_params = enhanced_model_results[0].get('Total_Parameters', 0)
            baseline_params = baseline_model_results[0].get('Total_Params', 0)
            
            if enhanced_params > 0 and baseline_params > 0:
                param_increase = enhanced_params - baseline_params
                param_increase_pct = (param_increase / baseline_params) * 100
                
                print(f"{model.upper():12}: Baseline={baseline_params:8,}, "
                      f"Enhanced={enhanced_params:8,}, "
                      f"Increase={param_increase:+8,} ({param_increase_pct:+.1f}%)")

def main():
    """메인 함수"""
    print("Enhanced vs Baseline Model Comparison")
    print("="*60)
    
    # 결과 로드
    enhanced_results, baseline_results = load_results()
    
    if not enhanced_results:
        print("⚠️  Enhanced 모델 결과를 찾을 수 없습니다.")
        print("   먼저 enhanced 모델들을 훈련해주세요:")
        print("   python seed_train.py --epochs 10 --seeds 24 --models resnet densenet")
    
    if not baseline_results:
        print("⚠️  Baseline 모델 결과를 찾을 수 없습니다.")
        print("   먼저 baseline 모델들을 훈련해주세요:")
        print("   python train_baseline.py --epochs 10 --seeds 24 --models resnet densenet")
        print("   결과는 'baseline_results' 폴더에 저장됩니다.")
    
    if enhanced_results and baseline_results:
        # 성능 비교 플롯 생성
        create_comparison_plot(enhanced_results, baseline_results)
        
        # 파라미터 효율성 분석
        analyze_parameter_efficiency(enhanced_results, baseline_results)
        
        # 결과 저장
        print(f"\n📊 Comparison plot saved as: model_comparison_enhanced_vs_baseline.png")
    
    else:
        print("\n📋 사용법:")
        print("1. Enhanced 모델 훈련:")
        print("   python seed_train.py --epochs 10 --seeds 24 --models resnet densenet")
        print("   결과는 'experiment_results_YYYYMMDD_HHMMSS' 폴더에 저장됩니다.")
        print("\n2. Baseline 모델 훈련:")
        print("   python train_baseline.py --epochs 10 --seeds 24 --models resnet densenet")
        print("   결과는 'baseline_results' 폴더에 저장됩니다.")
        print("\n3. 모델 비교:")
        print("   python compare_models.py")

if __name__ == "__main__":
    main()
