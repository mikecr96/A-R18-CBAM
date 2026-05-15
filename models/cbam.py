import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Implementation of the Channel Attention Module (CAM) as proposed in CBAM.
    The module performs adaptive feature recalibration by modeling inter-channel 
    dependencies via shared MLP architectures over pooled spatial descriptors.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        # Shared MLP structure using 1x1 convolutions for 4D tensor compatibility
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze-and-Excitation inspired aggregation of spatial context
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    Implementation of the Spatial Attention Module (SAM). 
    Focuses on 'where' the informative part is by exploiting spatial inter-dependencies
    through a large-kernel convolution over concatenated pooling maps.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Padding is set to maintain input dimensions for feature map integration
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class BasicBlockCBAM(nn.Module):
    """
    Residual Basic Block augmented with the Convolutional Block Attention Module.
    This architecture integrates sequential channel and spatial attention within
    the residual path to enhance high-cardinality feature extraction.
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False, ratio=16):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.use_cbam = use_cbam
        if self.use_cbam:
            self.ca = ChannelAttention(planes, ratio=ratio)
            self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.use_cbam:
            # Sequential Attention Refinement: Channel first, then Spatial
            out = self.ca(out) * out
            out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)