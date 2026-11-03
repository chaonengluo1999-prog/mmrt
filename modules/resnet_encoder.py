"""
MMRT Module 4: Visual Feature Extraction
ResNet-based visual backbone with spatial graph convolution for sign language recognition.
"""
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from einops import rearrange

from .gcn_lib.torch_vertex import Grapher, act_layer
from .gcn_lib.temgraph import TemporalGraph

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes,
        kernel_size=(1, 3, 3), stride=(1, stride, stride),
        padding=(0, 1, 1), bias=False)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# Keep internal class name for state_dict compatibility
BasicBlock = ResidualBlock


class MMRTVisualBackbone(nn.Module):
    """
    MMRT visual backbone: ResNet18 + spatial/temporal graph convolution.
    Module 4 in the MMRT architecture.
    """

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                               padding=(0, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.localG = Grapher(in_channels=256, kernel_size=3, dilation=1, conv='edge',
                              act='relu', norm="batch", bias=True, stochastic=False,
                              epsilon=0.0, r=1, n=14 * 14, drop_path=0.0, relative_pos=True)
        self.localG2 = Grapher(in_channels=512, kernel_size=4, dilation=1, conv='edge',
                               act='relu', norm="batch", bias=True, stochastic=False,
                               epsilon=0.0, r=1, n=7 * 7, drop_path=0.0, relative_pos=True)
        self.temporalG = TemporalGraph(k=14 * 14 // 4, in_channels=256, drop_path=0)
        self.temporalG2 = TemporalGraph(k=7 * 7, in_channels=512, drop_path=0)
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')
        x = x + self.localG(x) * self.alpha[0]
        x = x + self.temporalG(x, N) * self.alpha[1]
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)

        x = self.layer4(x)

        N, C, T, H, W = x.size()
        x = rearrange(x, 'N C T H W -> (N T) C H W')
        x = x + self.localG2(x) * self.alpha[2]
        x = x + self.temporalG2(x, N) * self.alpha[3]
        x = x.view(N, T, C, H, W).permute(0, 2, 1, 3, 4)

        x = x.transpose(1, 2).contiguous()
        x = x.view((-1,) + x.size()[2:])
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Keep original name for state_dict compatibility
ResNet = MMRTVisualBackbone


def build_visual_backbone(**kwargs):
    """Build MMRT visual backbone (ResNet18 + GCN)."""
    model = MMRTVisualBackbone(ResidualBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


# Alias for backward compatibility
resnet18 = build_visual_backbone
