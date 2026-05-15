import torch
import torch.nn as nn
from cbam import BasicBlockCBAM # Importing our modular attention blocks

class AsymmResNet18CBAM(nn.Module):
    """
    Modified ResNet-18 architecture with asymmetric strides for 
    Handwritten Digit Sequence Recognition (HDSR).
    The design optimizes for horizontal feature preservation, critical for
    multi-digit sequences in electoral auditing (PREP).
    """
    def __init__(self, cbam_layers=[False, False, True, True], ratio=16, num_classes=1001):
        super().__init__()
        self.inplanes = 64
        
        # Standard initial conv layer for high-level feature extraction
        # Efficient resolution reduction from 64x192 to 32x96
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stride 1 Maxpool to preserve topological information of the sequence
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Layer 1: Prioritizes horizontal resolution (stride 2, 1)
        self.layer1 = self._make_layer(64, 2, stride=(2, 1), use_cbam=cbam_layers[0], ratio=ratio)
        
        # Layer 2: Balanced downsampling (stride 2, 2)
        self.layer2 = self._make_layer(128, 2, stride=(2, 2), use_cbam=cbam_layers[1], ratio=ratio)
        
        # Layer 3: High-level abstraction with CBAM (r=16 as default)
        self.layer3 = self._make_layer(256, 2, stride=(2, 2), use_cbam=cbam_layers[2], ratio=ratio)
        
        # Layer 4: Violent asymmetric downsampling to collapse height (1x8)
        # Final refinement before Global Average Pooling (GAP)
        self.layer4 = self._make_layer(512, 2, stride=(4, 4), use_cbam=cbam_layers[3], ratio=ratio)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes) # Adjusted for 1,001 classes (high-cardinality)

    def _make_layer(self, planes, blocks, stride=1, use_cbam=False, ratio=16):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            # Identity projection for dimension matching in residual connections
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlockCBAM(self.inplanes, planes, stride, downsample, use_cbam=use_cbam, ratio=ratio)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlockCBAM(self.inplanes, planes, use_cbam=use_cbam, ratio=ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Computational path following the proposed asymmetric feature refinement
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))