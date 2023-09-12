import torch.nn as nn

class CoinClassifier(nn.Module):
    def _init_(self, num_classes=6):  # 6 different coin types
        super(CoinClassifier, self).__init__()
        
        self.encoder = nn.Sequential( #sirayla
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Input channels = 3 for RGB
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1),   # 4x4x2048
            nn.ReLU(),
        )
        
        self.dense_section = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * 4 * 4, 64),  # After 5 convolutional layers with stride 2, size becomes 16x16
            nn.ReLU(),
            nn.Linear(64, num_classes),  # After 5 convolutional layers with stride 2, size becomes 16x16
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.dense_section(x)
        return x