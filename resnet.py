from torch import nn
from torchvision import models

from config import NUM_CLASSES

model = models.resnet18(
    weights = models.ResNet18_Weights.DEFAULT
)

# Freeze all layers to keep basic feature detection intact
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze layer4 for new feature detection
for param in model.layer4.parameters():
    param.requires_grad = True

model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)
