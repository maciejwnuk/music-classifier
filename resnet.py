import torch
from torch import nn
from torchvision import models

from config import NUM_CLASSES

model = models.resnet18(
    weights = models.ResNet18_Weights.DEFAULT
)

# Replace 3D (RGB) convolution layer to 1D (grayscale)
conv = nn.Conv2d(
    in_channels  = 1,
    out_channels = model.conv1.out_channels,
    kernel_size  = model.conv1.kernel_size,         # pyright: ignore[reportArgumentType]
    stride       = model.conv1.stride,              # pyright: ignore[reportArgumentType]
    padding      = model.conv1.padding,             # pyright: ignore[reportArgumentType]
    dilation     = model.conv1.dilation,            # pyright: ignore[reportArgumentType]
    groups       = model.conv1.groups,
    bias         = (model.conv1.bias is not None),
    padding_mode = model.conv1.padding_mode
)

with torch.no_grad():
    conv.weight = nn.Parameter(
        torch.sum(
            model.conv1.weight.clone(),
            dim = 1, keepdim = True
        )
    )

model.conv1 = conv

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
