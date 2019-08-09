import torch.nn as nn
from torch import unsqueeze

class ConvNeuralNetwork(nn.Module):
  def __init__(self):
    super(ConvNeuralNetwork, self).__init__()
    self.layer1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(8),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    
    self.layer2 = nn.Sequential(
      nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2),
      nn.BatchNorm2d(16),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),
    )
    
    self.fc1 = nn.Sequential(
      nn.Dropout(),
      nn.Linear(32 * 32 * 16, 1024),
      nn.ReLU()
    )
    
    self.fc2 = nn.Linear(1024, 2)
          
  def forward(self, x):
    # Add the channels dimension to get a 4-channel input,
    # which PyTorch expects (BATCH_SIZE, CHANNELS, DIM1, DIM2)
    if len(x.shape) != 4:
      x = unsqueeze(x, 1)

    # Pass through the first two convolutional layers
    out = self.layer1(x)
    out = self.layer2(out)

    # Resize for the fully connected layers
    out = out.reshape(out.size(0), -1)

    # Pass through the fully connected layer
    out = self.fc1(out)
    out = self.fc2(out)

    return out
