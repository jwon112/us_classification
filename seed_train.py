import time
import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
from tqdm import tqdm
import wandb
from torchmetrics.classification import F1Score, Precision, Recall, ConfusionMatrix
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from functions_for_train import find_image_sizes, EarlyStopping, calculate_optimal_size, get_unique_filename, save_results_to_csv
from model import EnhancedResNet, CustomModel
from DenseNet_model import EnhancedDenseNet
from Grad_Cam import log_gradcam_examples, log_gradcam_to_wandb, overlay_heatmap_on_image, generate_gradcam_heatmap, visualize_cam, show_cam_on_image, GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()

# Set random seed for reproducibility

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# Argument Parser Setup
random_seeds = random.sample(range(1, 3000), 20)

parser = argparse.ArgumentParser(description='Train multiple models for US classification comparison.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training, set to -1 for auto allocation')
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights (.pth file)')
parser.add_argument('--seeds', nargs='+', type=int, default=[24], help='List of seeds to try for best result')
parser.add_argument('--models', nargs='+', type=str, default=['resnet', 'densenet'], 
                   choices=['resnet', 'densenet'], help='List of models to train')
args = parser.parse_args()

# WandB initialization
wandb.init(project="US_classification_model_comparison")
# wandb.config.update({"learning_rate": 6e-5, "epochs": args.epochs, "batch_size": args.batch_size})
wandb.config.update({"learning_rate": 0.0001 , "epochs": args.epochs, "batch_size": args.batch_size})
#previous lr : 7.316701740442333e-05

# FLOPs 계산을 위한 패키지 설치 안내
print("Note: For FLOPs calculation, install thop: pip install thop")
print("=" * 50)

# Load dataset and setup data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    # transforms.RandomRotation(15),      # -15도에서 15도 사이로 회전
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도, 색조 변경
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 프로젝트 루트 디렉토리 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 경로 설정 (base_dir 사용)
train_path = os.path.join(BASE_DIR, "DATA", "BUSI", "BUSI", "train")
test_path = os.path.join(BASE_DIR, "DATA", "BUSI", "BUSI", "test")

# 경로 존재 확인 및 출력 (디버깅용)
print(f"Base directory: {BASE_DIR}")
print(f"Train path: {train_path}")
print(f"Test path: {test_path}")
print(f"Train path exists: {os.path.exists(train_path)}")
print(f"Test path exists: {os.path.exists(test_path)}")

# 경로가 존재하지 않으면 에러 발생
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Train path does not exist: {train_path}")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test path does not exist: {test_path}")

# Create datasets with transforms
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

# Set batch size dynamically
batch_size = args.batch_size if args.batch_size != -1 else len(train_dataset) // 100

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Combine dataloaders into a dictionary
dataloaders = {'train': train_dataloader, 'test': test_dataloader}

precision_metric = Precision(num_classes=3, average='macro', task='multiclass').to(device)
recall_metric = Recall(num_classes=3, average='macro', task='multiclass').to(device)
confusion_metric = ConfusionMatrix(num_classes=3, task="multiclass").to(device)


def count_parameters(model):
    """모델의 파라미터 개수를 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """모델의 FLOPs를 계산"""
    try:
        from thop import profile
        input_tensor = torch.randn(input_size).to(device)  # GPU로 이동
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        
        print(f"FLOPs: {flops:,}")
        print(f"Parameters: {params:,}")
        
        return flops, params
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return None, None
    except RuntimeError as e:
        print(f"FLOPs calculation failed: {e}")
        return None, None


def calculate_model_size_mb(model):
    """모델 크기를 MB 단위로 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    print(f"Model size: {model_size_mb:.2f} MB")
    return model_size_mb


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct_predictions = 0.0, 0
    f1_metric = F1Score(num_classes=3, average='macro', task='multiclass').to(device)
    precision_metric = Precision(num_classes=3, average='macro', task='multiclass').to(device)
    recall_metric = Recall(num_classes=3, average='macro', task='multiclass').to(device)

    for inputs, labels in tqdm(dataloader, desc="Training", unit="batch"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        correct_predictions += torch.sum(preds == labels.data).item()

        f1_metric.update(outputs, labels)
        precision_metric.update(outputs, labels)
        recall_metric.update(outputs, labels)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct_predictions / len(dataloader.dataset)
    epoch_f1 = f1_metric.compute()
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()

    # Log metrics with wandb
    wandb.log({
        "Train Loss": epoch_loss,
        "Train Accuracy": epoch_acc,
        "Train F1": epoch_f1.item(),
        "Train Precision": epoch_precision.item(),
        "Train Recall": epoch_recall.item()
    })

    return epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall


def test(model, test_dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_preds = []
    all_labels = []

    # tqdm으로 진행 상태 표시
    with torch.no_grad():
        for inputs, labels in tqdm(test_dataloader, desc="Testing", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data).item()
            print(f"Batch Loss: {loss.item():.4f}, Precision: {precision_metric.compute():.4f}, Recall: {recall_metric.compute():.4f}")


    test_loss = running_loss / len(test_dataloader.dataset)
    test_acc = correct_predictions / len(test_dataloader.dataset)

    # Compute metrics
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print("Confusion Matrix:\n", cm)

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Benign", "Malignant"])
    disp.plot(cmap="viridis")

    # Show and auto-close plot
    plt.pause(10)  # Display the plot for 10 seconds
    plt.close()    # Automatically close the plot after 10 seconds

    # Log metrics with wandb
    wandb.log({
        "Test Loss": test_loss,
        "Test Accuracy": test_acc,
        "Test F1": test_f1,
        "Test Precision": test_precision,
        "Test Recall": test_recall
    })

    return test_loss, test_acc, test_f1, test_precision, test_recall



def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=25, model_name="", seed=0):
    results = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_test_acc, best_epoch_acc = 0.0, 0
    best_epoch_confusion_matrix = None
    best_f1, best_precision, best_recall = 0.0, 0.0, 0.0  # 추가


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Initialize metrics storage
        epoch_metrics = {
            'train_loss': 0.0, 'train_acc': 0.0, 'train_f1': 0.0, 'train_precision': 0.0, 'train_recall': 0.0,
            'test_loss': 0.0, 'test_acc': 0.0, 'test_f1': 0.0, 'test_precision': 0.0, 'test_recall': 0.0
        }

        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, correct_predictions = 0.0, 0
            f1_metric = F1Score(num_classes=3, average='macro', task='multiclass').to(device)
            precision_metric = Precision(num_classes=3, average='macro', task='multiclass').to(device)
            recall_metric = Recall(num_classes=3, average='macro', task='multiclass').to(device)

            all_preds, all_labels = [], []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch + 1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    f1_metric.update(outputs, labels)
                    precision_metric.update(outputs, labels)
                    recall_metric.update(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)
                    correct_predictions += torch.sum(preds == labels.data).item()

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = correct_predictions / len(dataloaders[phase].dataset)
            epoch_f1 = f1_metric.compute()
            epoch_precision = precision_metric.compute()
            epoch_recall = recall_metric.compute()

            # Save epoch metrics
            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc
            epoch_metrics[f'{phase}_f1'] = epoch_f1.item()
            epoch_metrics[f'{phase}_precision'] = epoch_precision.item()
            epoch_metrics[f'{phase}_recall'] = epoch_recall.item()

            if phase == 'test':
                
                # Compute and log Confusion Matrix
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
                print(f"Confusion Matrix for Epoch {epoch + 1}:\n", cm)

                # Plot Confusion Matrix
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Benign", "Malignant"])
                disp.plot(cmap=plt.cm.Blues)

                # Save the plot to experiment folder and WandB
                confusion_matrix_file = os.path.join(experiment_dir, f"confusion_matrix_{model_name}_seed_{seed}_epoch_{epoch + 1}.png")
                plt.savefig(confusion_matrix_file)
                wandb.log({
                    f"{model_name.upper()} Confusion Matrix Epoch {epoch + 1}": wandb.Image(confusion_matrix_file)
                })
                plt.close()

                # Track the best accuracy and Confusion Matrix
                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_epoch_acc = epoch + 1
                    best_epoch_confusion_matrix = cm

                    best_f1 = epoch_metrics['test_f1']  # 추가
                    best_precision = epoch_metrics['test_precision']  # 추가
                    best_recall = epoch_metrics['test_recall']  # 추가

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
                  f"F1 Score: {epoch_f1:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}")

            # Log to WandB
            wandb.log({
                f"{model_name.upper()}_{phase.capitalize()}_Loss": epoch_loss,
                f"{model_name.upper()}_{phase.capitalize()}_Accuracy": epoch_acc,
                f"{model_name.upper()}_{phase.capitalize()}_F1": epoch_f1.item(),
                f"{model_name.upper()}_{phase.capitalize()}_Precision": epoch_precision.item(),
                f"{model_name.upper()}_{phase.capitalize()}_Recall": epoch_recall.item()
            })

        results.append([
            epoch + 1,
            epoch_metrics['train_loss'], epoch_metrics['train_acc'], epoch_metrics['train_f1'],
            epoch_metrics['train_precision'], epoch_metrics['train_recall'],
            epoch_metrics['test_loss'], epoch_metrics['test_acc'], epoch_metrics['test_f1'],
            epoch_metrics['test_precision'], epoch_metrics['test_recall']
        ])

        # 시드별 결과를 실험 폴더에 저장
        seed_results_file = os.path.join(experiment_dir, f"results_{model_name}_seed_{seed}.csv")
        save_results_to_csv(results, seed_results_file)

        if early_stopping(epoch_metrics['test_loss'], model):
            break

    # Print best epoch and corresponding confusion matrix
    print(f"\nBest Test Accuracy: {best_test_acc:.4f} at Epoch {best_epoch_acc}")
    print(f"Confusion Matrix for Best Epoch:\n{best_epoch_confusion_matrix}")

    # Return only the values required
    return best_test_acc, best_epoch_acc, best_epoch_confusion_matrix, epoch_f1.item(), epoch_precision.item(), epoch_recall.item()


def log_confusion_matrix_to_wandb(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="viridis", ax=ax)
    plt.title("Confusion Matrix")
    
    # Log to WandB
    wandb.log({"Confusion Matrix": wandb.Image(fig)})
    plt.close(fig)

# Main loop for training with multiple seeds
best_seed, best_accuracy = None, 0.0
columns = ["Seed", "Total Parameters", "Trainable Parameters", "FLOPs", "Model Size (MB)", "Test Accuracy", "Best Epoch", "Best F1 Score", "Test Precision", "Test Recall"]
seed_results = []

# Run training for each model and seed
if __name__ == "__main__":
    best_seed, best_accuracy, best_model_name = None, 0.0, None
    all_results = []
    
    # 실행별 결과 폴더 생성
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    experiment_dir = f"experiment_results_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Results will be saved in: {experiment_dir}")
    print(f"Training models: {args.models}")
    print("=" * 50)

    # 각 모델에 대해 훈련
    for model_name in args.models:
        print(f"\n{'='*20} Training {model_name.upper()} Model {'='*20}")
        
        for seed in args.seeds:
            print(f"\nTraining {model_name.upper()} with Seed: {seed}")
            set_seed(seed)  # Set reproducible seed

            # 모델 선택
            if model_name == 'resnet':
                model = CustomModel(num_classes=3).to(device)
            elif model_name == 'densenet':
                model = EnhancedDenseNet(num_classes=3).to(device)
            else:
                print(f"Unknown model: {model_name}, skipping...")
                continue
            
            # 모델 정보 출력 및 계산
            print(f"\n=== {model_name.upper()} Model Information for Seed {seed} ===")
            total_params, trainable_params = count_parameters(model)
            flops, _ = calculate_flops(model)
            model_size_mb = calculate_model_size_mb(model)
            print("=" * 50)
            
            # 모델 정보를 WandB에 로깅
            wandb.log({
                f"{model_name}_seed_{seed}_total_parameters": total_params,
                f"{model_name}_seed_{seed}_trainable_parameters": trainable_params,
                f"{model_name}_seed_{seed}_flops": flops if flops else 0,
                f"{model_name}_seed_{seed}_model_size_mb": model_size_mb
            })
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.NAdam(model.parameters(), lr=wandb.config["learning_rate"])

            # Collect 6 values for each seed
            test_accuracy, best_epoch_acc, best_epoch_confusion_matrix, best_f1, best_precision, best_recall = train_and_evaluate_model(
                model, dataloaders, criterion, optimizer, num_epochs=args.epochs, model_name=model_name, seed=seed
            )

            # Append all results for the model and seed (including model information)
            all_results.append([model_name, seed, total_params, trainable_params, flops if flops else 0, model_size_mb, test_accuracy, best_epoch_acc, best_f1, best_precision, best_recall])
            
            # Update best results
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_seed = seed
                best_model_name = model_name


    # Unpack correctly with 11 variables (including model name and model information)
    for model_name, seed, total_params, trainable_params, flops, model_size_mb, test_acc, best_epoch, best_f1, test_precision, test_recall in all_results:
        print(f"Model: {model_name.upper()}, Seed: {seed}, Params: {total_params:,}, FLOPs: {flops:,}, Size: {model_size_mb:.2f}MB, "
              f"Test Acc: {test_acc:.4f}, Best Epoch: {best_epoch}, "
              f"F1: {best_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    if best_seed is not None:
        print(f"\nLogging Confusion Matrix for Best Model: {best_model_name.upper()} with Seed: {best_seed}")
        set_seed(best_seed)  # Best Seed로 재현성 확보

        # 모델 재초기화 및 테스트 데이터로 Confusion Matrix 생성
        if best_model_name == 'resnet':
            best_model = CustomModel(num_classes=3).to(device)
        elif best_model_name == 'densenet':
            best_model = EnhancedDenseNet(num_classes=3).to(device)
        
        # Best seed 모델 정보도 출력
        print(f"\n=== Best Model Information ({best_model_name.upper()}, Seed {best_seed}) ===")
        best_total_params, best_trainable_params = count_parameters(best_model)
        best_flops, _ = calculate_flops(best_model)
        best_model_size_mb = calculate_model_size_mb(best_model)
        print("=" * 50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.NAdam(best_model.parameters(), lr=wandb.config["learning_rate"])
        
        # Train the model to load weights of best seed if necessary (skip training here if preloaded)
        _, best_dataloader = dataloaders['test'], test_dataloader  # 선택적으로 테스트 데이터만 활용
        best_model.eval()

        # Confusion Matrix 생성
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(best_dataloader, desc="Generating Confusion Matrix"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Log Confusion Matrix to WandB
        log_confusion_matrix_to_wandb(all_labels, all_preds, class_names=["Normal", "Benign", "Malignant"])
    
    # Print a summary of best results for each model and seed
    print("\nBest Results per Model and Seed:")
    print("Model comparison results:")

    # 최종 결과를 실험 폴더에 저장
    columns = ['Model', 'Seed', 'Total_Params', 'Trainable_Params', 'FLOPs', 'Model_Size_MB', 
               'Test_Accuracy', 'Best_Epoch', 'F1_Score', 'Precision', 'Recall']
    results_df = pd.DataFrame(all_results, columns=columns)
    
    # 모델별로 결과 분리하여 저장
    for model_name in args.models:
        model_results = results_df[results_df['Model'] == model_name]
        if not model_results.empty:
            model_results_file = os.path.join(experiment_dir, f"{model_name}_results_summary.csv")
            model_results.to_csv(model_results_file, index=False)
            print(f"{model_name.upper()} results saved to: {model_results_file}")
    
    # 전체 비교 결과 저장
    overall_results_file = os.path.join(experiment_dir, f"overall_model_comparison_results.csv")
    results_df.to_csv(overall_results_file, index=False)
    
    # WandB에도 저장 (Windows 권한 문제 방지)
    try:
        # Windows에서는 파일을 직접 복사하는 방식 사용
        wandb.save(overall_results_file, base_path=experiment_dir)
        print("Results also saved to WandB successfully")
    except OSError as e:
        print(f"Warning: Could not save to WandB due to permission issue: {e}")
        print("Results are still saved locally in the experiment folder")
        # 대안: 파일 내용을 WandB에 직접 로깅
        try:
            with open(overall_results_file, 'r') as f:
                csv_content = f.read()
            wandb.log({"overall_results_csv": wandb.Html(f"<pre>{csv_content}</pre>")})
            print("Results logged to WandB as HTML content")
        except Exception as e2:
            print(f"Could not log results to WandB: {e2}")
    
    print(f"\nTraining complete. All results saved in: {experiment_dir}")
    print(f"Overall results: {overall_results_file}")
    print("=" * 50)
    
    wandb.finish()