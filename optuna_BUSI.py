import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import optuna
from model import CustomModel

# 시드 설정
torch.manual_seed(24)
torch.cuda.manual_seed_all(24)
np.random.seed(24)

# 경로 설정
# train_path = "BUSI/BUSI/train"
# test_path = "BUSI/BUSI/test"
train_path = "GDPHSYSUCC/GDPHnSYSUCC_all/train"
# train_path = "GDPHSYSUCC/GDPHnSYSUCC/GDPH/train"
# train_path = "GDPHSYSUCC/GDPHnSYSUCC/SYSUCC/train"

test_path = "GDPHSYSUCC/GDPHnSYSUCC_all/test"
# test_path = "GDPHSYSUCC/GDPHnSYSUCC/GDPH/test"
# test_path = "GDPHSYSUCC/GDPHnSYSUCC/SYSUCC/test"
# 데이터 변환
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

# 데이터 로더
train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 모델 정의 (예제 모델, CustomModel 사용)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    lr = trial.suggest_loguniform("lr", 7e-5, 10e-5)
    model = CustomModel(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 100
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
            
        acc = accuracy_score(all_labels, all_preds)
        if acc > best_acc:
            best_acc = acc
    
    return best_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# 최적의 하이퍼파라미터 적용
best_lr = study.best_params["lr"]
# model = CustomModel(num_classes=3).to(device)
model = CustomModel(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=best_lr)

# 최종 Test 성능 평가
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels_np)

final_acc = accuracy_score(all_labels, all_preds)
final_precision = precision_score(all_labels, all_preds, average='macro')
final_recall = recall_score(all_labels, all_preds, average='macro')
final_f1 = f1_score(all_labels, all_preds, average='macro')
final_cm = confusion_matrix(all_labels, all_preds)

print(f"Test Results | Acc: {final_acc:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f} | F1-score: {final_f1:.4f}")

# Confusion Matrix 저장
plt.figure(figsize=(6,6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
