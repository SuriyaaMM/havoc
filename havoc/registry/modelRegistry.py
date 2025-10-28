import torch
import torch.nn as nn

# Model Classifier Linear Version 1
class MCLv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 120)
        )

    def forward(self, x):
        return self.ff(x)

# Model Classifier CNN Version 1
class MCCv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128*4*32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 120)
        )

    def forward(self, x):
        x = self.ff(x)
        return x.view(-1, 12, 10)

# Model Classifier CNN Version 2
class MCCv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.Flatten(),
            nn.Linear(128*4*32, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 120)
        )

    def forward(self, x):
        x = self.ff(x)
        return x.view(-1, 12, 10)
    
# Model Classifier CNN Version 3 
# Same as 2, but separated CNN & MLP Backbone for Gradient History Logging
class MCCv3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(),
        )
        self.fmlp =nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*32, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 120)
        )

    def forward(self, x):
        x = self.fcnn(x)
        x = self.fmlp(x)
        return x.view(-1, 12, 10)
    
# Model Classifier CNN + LSTM Version 1
# NOTE: outputs raw logits
class MCCLv1(nn.Module):
    def __init__(self):
        super().__init__()

        # input: [B, 1, 32, 256]
        self.cnnNetwork = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # output: [B, 128, 4, 32]

        # collapse along height & channel dimension and keep intact width dimension
        # so it appears to read from left -> right & right -> left (bi-directional lstm)

        # input: [B, 32, 512]
        self.lstmNetwork = nn.LSTM(
            input_size=512,      
            hidden_size=256,     
            num_layers=2,        
            bidirectional=True,  
            batch_first=True,   
            dropout=0.2      
        )
        # output: [B, T, hidden_size*2]
        # we need 11 output classes (10 digits (0-9) + 1 blank token for CTC)
        self.classifier = nn.Linear(512, 11)

    def forward(self, x: torch.Tensor):  
        xx: torch.Tensor = self.cnnNetwork(x)
        B, C, H, W = xx.shape
        xxx = xx.view(B, C * H, W)
        xxxx = xxx.permute(0, 2, 1)  
        xxxxx, (_, _) = self.lstmNetwork(xxxx)
        return self.classifier(xxxxx)