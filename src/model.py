import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), pool=(2,2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=(kernel[0]//2, kernel[1]//2))
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.pool(x)
        return x

class DeepfakeCNN(nn.Module):
    def __init__(self, n_mels=64):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(1, 16, kernel=(5,5), pool=(2,2)),
            ConvBlock(16, 32, kernel=(3,3), pool=(2,2)),
            ConvBlock(32, 64, kernel=(3,3), pool=(2,2)),
            ConvBlock(64, 128, kernel=(3,3), pool=(2,1)),  # keep more time resolution
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B,1,n_mels,T)
        z = self.enc(x)
        z = self.global_pool(z)
        z = z.view(z.size(0), -1)
        logits = self.fc(z).squeeze(1)
        return logits
