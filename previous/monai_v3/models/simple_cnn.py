# simple_cnn.py
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Flatten(start_dim=1),
            nn.Linear(64 * 32 * 32, 128),  # For 256x256 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = self.layers(x)
        assert logits.ndim == 2, f"Expected [B, num_classes], got {logits.shape}"
        print(f"[SimpleCNN] logits shape: {logits.shape}")  # Optional: comment out in production
        return logits


# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels=1, num_classes=2):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(64 * 32 * 32, 128)  # For 256x256 input
#         self.fc2 = nn.Linear(128, num_classes)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = torch.flatten(x, 1)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
#         return x
