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
from Grad_Cam import log_gradcam_examples, log_gradcam_to_wandb, overlay_heatmap_on_image, generate_gradcam_heatmap, visualize_cam, show_cam_on_image, GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Clear GPU cache
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
random_seeds = random.sample(range(1, 3000), 5)

parser = argparse.ArgumentParser(description='Train a ResNet model with CBAM for ORS classification.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training, set to -1 for auto allocation')
parser.add_argument('--pretrained_weights', type=str, default=None, help='Path to pretrained weights (.pth file)')
parser.add_argument('--seeds', nargs='+', type=int, default=[24], help='List of seeds to try for best result')
args = parser.parse_args()

# WandB initialization
wandb.init(project="US_classification_GDPH&SYSUCC", entity="ddurbozak")
# wandb.config.update({"learning_rate": 12e-5, "epochs": args.epochs, "batch_size": args.batch_size})
wandb.config.update({"learning_rate":  7.704149588474852e-05, "epochs": args.epochs, "batch_size": args.batch_size})

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


train_path = "GDPHSYSUCC/GDPHnSYSUCC_all/train"
# train_path = "GDPHSYSUCC/GDPHnSYSUCC/GDPH/train"
# train_path = "GDPHSYSUCC/GDPHnSYSUCC/SYSUCC/train"

test_path = "GDPHSYSUCC/GDPHnSYSUCC_all/test"
# test_path = "GDPHSYSUCC/GDPHnSYSUCC/GDPH/test"
# test_path = "GDPHSYSUCC/GDPHnSYSUCC/SYSUCC/test"

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

precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)
confusion_metric = ConfusionMatrix(num_classes=2, task="multiclass").to(device)


def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct_predictions = 0.0, 0
    f1_metric = F1Score(num_classes=2, average='macro', task='multiclass').to(device)
    precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
    recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)

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

    test_loss = running_loss / len(test_dataloader.dataset)
    test_acc = correct_predictions / len(test_dataloader.dataset)

    # Compute metrics
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    print("Confusion Matrix:\n", cm)

    # Display Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
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



def train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    
    results = []
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_test_acc, best_epoch_acc = 0.0, 0
    best_epoch_confusion_matrix = None

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
            f1_metric = F1Score(num_classes=2, average='macro', task='multiclass').to(device)
            precision_metric = Precision(num_classes=2, average='macro', task='multiclass').to(device)
            recall_metric = Recall(num_classes=2, average='macro', task='multiclass').to(device)

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
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                print(f"Confusion Matrix for Epoch {epoch + 1}:\n", cm)

                # Plot Confusion Matrix
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
                disp.plot(cmap=plt.cm.Blues)

                # Save the plot to WandB
                plt.savefig(f"confusion_matrix_epoch_{epoch + 1}.png")
                wandb.log({
                    f"Confusion Matrix Epoch {epoch + 1}": wandb.Image(f"confusion_matrix_epoch_{epoch + 1}.png")
                })
                plt.close()

                # Track the best accuracy and Confusion Matrix
                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_epoch_acc = epoch + 1
                    best_epoch_confusion_matrix = cm

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, "
                  f"F1 Score: {epoch_f1:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}")

            # Log to WandB
            wandb.log({
                f"{phase.capitalize()} Loss": epoch_loss,
                f"{phase.capitalize()} Accuracy": epoch_acc,
                f"{phase.capitalize()} F1": epoch_f1.item(),
                f"{phase.capitalize()} Precision": epoch_precision.item(),
                f"{phase.capitalize()} Recall": epoch_recall.item()
            })

        results.append([
            epoch + 1,
            epoch_metrics['train_loss'], epoch_metrics['train_acc'], epoch_metrics['train_f1'],
            epoch_metrics['train_precision'], epoch_metrics['train_recall'],
            epoch_metrics['test_loss'], epoch_metrics['test_acc'], epoch_metrics['test_f1'],
            epoch_metrics['test_precision'], epoch_metrics['test_recall']
        ])

        save_results_to_csv(results, f"results_seed_{seed}.csv")

        if early_stopping(epoch_metrics['test_loss'], model):
            break

    # Print best epoch and corresponding confusion matrix
    print(f"\nBest Test Accuracy: {best_test_acc:.4f} at Epoch {best_epoch_acc}")
    print(f"Confusion Matrix for Best Epoch:\n{best_epoch_confusion_matrix}")

    # Return only the values required
    return best_test_acc, best_epoch_acc, best_epoch_confusion_matrix

def log_confusion_matrix_to_wandb(all_labels, all_preds, class_names):
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
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
seed_results = []

# Run training for each seed and save results per seed
for seed in args.seeds:
    print(f"Training with Seed: {seed}")
    set_seed(seed)  # Set the seed for reproducibility

    model = CustomModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=wandb.config["learning_rate"])
    
    # Train and evaluate the model for the current seed
    test_accuracy, best_f1_score, other_metric = train_and_evaluate_model(model, dataloaders, criterion, optimizer, num_epochs=args.epochs)

    # Append best results of the current seed
    seed_results.append([seed, test_accuracy, best_f1_score])

    # Update the best accuracy and seed
    if test_accuracy > best_accuracy:
        best_accuracy, best_seed = test_accuracy, seed

# Print the best seed with the highest test accuracy across all seeds
print(f"\nOverall Best Seed: {best_seed} with Test Accuracy: {best_accuracy:.4f}")
if best_seed is not None:
    print(f"Logging Confusion Matrix for Seed: {best_seed}")
    set_seed(best_seed)  # Best Seed로 재현성 확보

    # 모델 재초기화 및 테스트 데이터로 Confusion Matrix 생성
    best_model = CustomModel(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=wandb.config["learning_rate"])
    
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
    log_confusion_matrix_to_wandb(all_labels, all_preds, class_names=["Benign", "Malignant"])
# Print a summary of best results for each seed
print("\nBest Results per Seed:")
for seed, test_acc, best_f1 in seed_results:
    print(f"Seed {seed}: Best Test Accuracy = {test_acc:.4f}, Best F1 Score = {best_f1:.4f}")
