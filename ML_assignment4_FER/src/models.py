import torch
import torch.nn as nn

# 1. TinyCNN (Run 1)
class TinyCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 24 * 24, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

# 2. SmallCNN (Run 2)
class SmallCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 12 * 12, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

# 3. DeepCNN_Overfit (Run 3)
class DeepCNN_Overfit(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 12 * 12, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# 4. OptimizedCNN (Run 4)
class OptimizedCNN(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout2d(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 12 * 12, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

# 5. ResNetStyleCNN (Run 5)
class ResNetStyleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Residual Block
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 24 * 24, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # Skip connection
        residual = x
        x = self.bn2(self.conv2(x))
        x += residual  # ბლოკის მიმატება
        x = self.relu(x)
        x = self.pool(x)
        return self.classifier(x)
