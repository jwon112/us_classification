import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from visualization import create_model_comparison_chart, create_parameter_efficiency_analysis, create_parameter_efficiency_charts, create_flops_efficiency_charts

def load_results():
    """결과 파일들을 로드하여 비교 데이터 생성"""
    
    # Enhanced 모델 결과 로드 (experiments/ 폴더에서 integrated_experiment_results_*)
    enhanced_results = []
    experiments_dir = "experiments"
    
    if os.path.exists(experiments_dir):
        # integrated_experiment_results_* 폴더 찾기
        integrated_dirs = [d for d in os.listdir(experiments_dir) if d.startswith('integrated_experiment_results_')]
        
        if integrated_dirs:
            # 가장 최근 통합 실험 결과 사용
            latest_dir = max(integrated_dirs)
            latest_path = os.path.join(experiments_dir, latest_dir)
            print(f"Loading integrated experiment results from: {latest_path}")
            
            # integrated_experiment_results.csv 찾기
            integrated_csv_path = os.path.join(latest_path, "integrated_experiment_results.csv")
            if os.path.exists(integrated_csv_path):
                integrated_df = pd.read_csv(integrated_csv_path)
                # enhanced 모델 결과만 필터링
                enhanced_df = integrated_df[integrated_df['model_type'] == 'enhanced']
                enhanced_results = enhanced_df.to_dict('records')
                print(f"   Loaded: integrated_experiment_results.csv ({len(enhanced_results)} enhanced entries)")
            else:
                print(f"   Error: integrated_experiment_results.csv not found in {latest_path}")
        else:
            # 기존 experiment_results_* 폴더에서 찾기 (fallback)
            experiment_dirs = [d for d in os.listdir(experiments_dir) if d.startswith('experiment_results_')]
            if experiment_dirs:
                latest_dir = max(experiment_dirs)
                latest_path = os.path.join(experiments_dir, latest_dir)
                print(f"   Fallback: Loading from experiment_results_* directory: {latest_path}")
                
                # overall_model_comparison_results.csv 찾기
                overall_csv_path = os.path.join(latest_path, "overall_model_comparison_results.csv")
                if os.path.exists(overall_csv_path):
                    enhanced_df = pd.read_csv(overall_csv_path)
                    enhanced_results = enhanced_df.to_dict('records')
                    print(f"   Loaded: overall_model_comparison_results.csv ({len(enhanced_results)} entries)")
                else:
                    # 다른 CSV 파일들 중에서 찾기
                    csv_files = [f for f in os.listdir(latest_path) if f.endswith('.csv')]
                    if csv_files:
                        print(f"   Warning: overall_model_comparison_results.csv not found")
                        print(f"   Available CSV files: {csv_files}")
                        # 첫 번째 CSV 파일 로드
                        enhanced_df = pd.read_csv(os.path.join(latest_path, csv_files[0]))
                        enhanced_results = enhanced_df.to_dict('records')
                        print(f"   Loaded: {csv_files[0]} ({len(enhanced_results)} entries)")
            else:
                print(f"   No integrated_experiment_results_* or experiment_results_* directories found in {experiments_dir}")
    else:
        print(f"   {experiments_dir} directory not found")
    
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
    
    # visualization 모듈의 함수 사용
    comparison_data = create_model_comparison_chart(
        enhanced_results, baseline_results, 
        save_path='model_comparison_enhanced_vs_baseline.png'
    )
    
    if comparison_data:
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
    
    # visualization 모듈의 함수 사용
    create_parameter_efficiency_analysis(enhanced_results, baseline_results)

def create_parameter_efficiency_charts_from_results(enhanced_results, baseline_results):
    """Enhanced와 Baseline 결과에서 파라미터 효율성 차트 생성"""
    
    if not enhanced_results or not baseline_results:
        print("⚠️  결과 데이터가 부족하여 파라미터 효율성 차트를 생성할 수 없습니다.")
        return
    
    # 데이터를 DataFrame으로 변환
    enhanced_df = pd.DataFrame(enhanced_results)
    baseline_df = pd.DataFrame(baseline_results)
    
    # 컬럼명 통일
    enhanced_df['model_type'] = 'enhanced'
    enhanced_df['model_name'] = enhanced_df['model_name']
    enhanced_df['final_accuracy'] = enhanced_df['final_accuracy']
    enhanced_df['total_params'] = enhanced_df['total_params']
    enhanced_df['flops'] = enhanced_df['flops']
    
    baseline_df['model_type'] = 'baseline'
    baseline_df['model_name'] = baseline_df['Model']
    baseline_df['final_accuracy'] = baseline_df['Test_Accuracy']
    baseline_df['total_params'] = baseline_df['Total_Params']
    baseline_df['flops'] = baseline_df['FLOPs']
    
    # 통합 DataFrame 생성
    combined_df = pd.concat([
        enhanced_df[['model_type', 'model_name', 'final_accuracy', 'total_params', 'flops']],
        baseline_df[['model_type', 'model_name', 'final_accuracy', 'total_params', 'flops']]
    ], ignore_index=True)
    
    # visualization 모듈의 함수 사용
    create_parameter_efficiency_charts(combined_df, ".")


def create_flops_efficiency_charts_from_results(enhanced_results, baseline_results):
    """Enhanced와 Baseline 결과에서 FLOPs 효율성 차트 생성"""
    
    if not enhanced_results or not baseline_results:
        print("⚠️  결과 데이터가 부족하여 FLOPs 효율성 차트를 생성할 수 없습니다.")
        return
    
    # 데이터를 DataFrame으로 변환
    enhanced_df = pd.DataFrame(enhanced_results)
    baseline_df = pd.DataFrame(baseline_results)
    
    # 컬럼명 통일
    enhanced_df['model_type'] = 'enhanced'
    enhanced_df['model_name'] = enhanced_df['model_name']
    enhanced_df['final_accuracy'] = enhanced_df['final_accuracy']
    enhanced_df['total_params'] = enhanced_df['total_params']
    enhanced_df['flops'] = enhanced_df['flops']
    
    baseline_df['model_type'] = 'baseline'
    baseline_df['model_name'] = baseline_df['Model']
    baseline_df['final_accuracy'] = baseline_df['Test_Accuracy']
    baseline_df['total_params'] = baseline_df['Total_Params']
    baseline_df['flops'] = baseline_df['FLOPs']
    
    # 통합 DataFrame 생성
    combined_df = pd.concat([
        enhanced_df[['model_type', 'model_name', 'final_accuracy', 'total_params', 'flops']],
        baseline_df[['model_type', 'model_name', 'final_accuracy', 'total_params', 'flops']]
    ], ignore_index=True)
    
    # visualization 모듈의 함수 사용
    create_flops_efficiency_charts(combined_df, ".")


def main():
    """메인 함수"""
    print("Enhanced vs Baseline Model Comparison")
    print("="*60)
    
    # 결과 로드
    enhanced_results, baseline_results = load_results()
    
    if not enhanced_results:
        print("⚠️  Enhanced 모델 결과를 찾을 수 없습니다.")
        print("   먼저 integrated experiment를 실행해주세요:")
        print("   python integrated_experiment.py --epochs 10 --seeds 24")
    
    if not baseline_results:
        print("⚠️  Baseline 모델 결과를 찾을 수 없습니다.")
        print("   먼저 baseline 모델들을 훈련해주세요:")
        print("   python train_baseline.py --epochs 100")
        print("   결과는 'baseline_results' 폴더에 저장됩니다.")
    
    if enhanced_results and baseline_results:
        # 성능 비교 플롯 생성
        create_comparison_plot(enhanced_results, baseline_results)
        
        # 파라미터 효율성 분석 (텍스트)
        analyze_parameter_efficiency(enhanced_results, baseline_results)
        
        # 파라미터 효율성 차트 생성
        print("\n📊 Creating parameter efficiency charts...")
        create_parameter_efficiency_charts_from_results(enhanced_results, baseline_results)
        
        # FLOPs 효율성 차트 생성
        print("\n📊 Creating FLOPs efficiency charts...")
        create_flops_efficiency_charts_from_results(enhanced_results, baseline_results)
        
        # 결과 저장
        print(f"\n📊 Comparison plot saved as: model_comparison_enhanced_vs_baseline.png")
        print(f"📊 Parameter efficiency charts saved as: parameter_efficiency_*.png")
        print(f"📊 FLOPs efficiency charts saved as: flops_efficiency_*.png")
    
    else:
        print("\n📋 사용법:")
        print("1. Baseline 모델 훈련 (한 번만):")
        print("   python train_baseline.py --epochs 100")
        print("   결과는 'baseline_results' 폴더에 저장됩니다.")
        print("\n2. Enhanced 모델 통합 실험:")
        print("   python integrated_experiment.py --epochs 100 --seeds 24 42 123")
        print("   결과는 'experiments/integrated_experiment_results_*' 폴더에 저장됩니다.")
        print("\n3. 모델 비교:")
        print("   python compare_models.py")

if __name__ == "__main__":
    main()