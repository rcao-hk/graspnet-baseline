from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = [
    'ResNet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d',
]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def load_weights_sequential(target, source_state):
    new_dict = OrderedDict()
    for (k1, _), (_, v2) in zip(target.state_dict().items(), source_state.items()):
        new_dict[k1] = v2
    target.load_state_dict(new_dict)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=1,
        groups=1,
        base_width=64,
    ):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")

        self.conv1 = conv3x3(
            inplanes, planes, stride=stride, dilation=dilation, groups=1
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(
            planes, planes, stride=1, dilation=dilation, groups=1
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        dilation=1,
        groups=1,
        base_width=64,
    ):
        super(Bottleneck, self).__init__()

        # ResNet:
        #   groups=1, base_width=64
        # ResNeXt:
        #   e.g. groups=32, base_width=4 or 8
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(
            width, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers=(3, 4, 23, 3),
        num_classes=1000,
        fully_conv=False,
        remove_avg_pool_layer=True,
        output_stride=32,
        groups=1,
        width_per_group=64,
    ):
        super(ResNet, self).__init__()

        self.output_stride = output_stride
        self.current_stride = 4
        self.current_dilation = 1
        self.remove_avg_pool_layer = remove_avg_pool_layer

        self.inplanes = 64
        self.fully_conv = fully_conv
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 保持你原来的 stride / dilation 逻辑
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if self.fully_conv:
            self.avgpool = nn.AvgPool2d(7, padding=3, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # 达到目标 output_stride 后，用 dilation 替代 stride
            if self.current_stride == self.output_stride:
                self.current_dilation = self.current_dilation * stride
                stride = 1
            else:
                self.current_stride = self.current_stride * stride

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dilation=self.current_dilation,
                groups=self.groups,
                base_width=self.base_width,
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    dilation=self.current_dilation,
                    groups=self.groups,
                    base_width=self.base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)      # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)    # /4

        x = self.layer1(x)
        x = self.layer2(x)     # /8 (unless output_stride logic changes it)
        x_3 = self.layer3(x)   # usually still /8 in your setup
        x32s = self.layer4(x_3)

        x = x32s

        if not self.remove_avg_pool_layer:
            x = self.avgpool(x)

        if not self.fully_conv:
            x = x.view(x.size(0), -1)

        # xfc = self.fc(x)

        return x32s, x_3


def _load_pretrained(model, arch):
    print(f"loading {arch} pretrained mdl.")
    state_dict = model_zoo.load_url(model_urls[arch])
    model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model = _load_pretrained(model, 'resnet18')
    return model


def resnet34(pretrained=False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model = _load_pretrained(model, 'resnet34')
    return model


def resnet50(pretrained=False):
    model = ResNet(
        Bottleneck, [3, 4, 6, 3],
        groups=1, width_per_group=64
    )
    if pretrained:
        model = _load_pretrained(model, 'resnet50')
    return model


def resnet101(pretrained=False):
    model = ResNet(
        Bottleneck, [3, 4, 23, 3],
        groups=1, width_per_group=64
    )
    if pretrained:
        model = _load_pretrained(model, 'resnet101')
    return model


def resnet152(pretrained=False):
    model = ResNet(
        Bottleneck, [3, 8, 36, 3],
        groups=1, width_per_group=64
    )
    if pretrained:
        model = _load_pretrained(model, 'resnet152')
    return model


def resnext50_32x4d(pretrained=False):
    model = ResNet(
        Bottleneck, [3, 4, 6, 3],
        groups=32, width_per_group=4
    )
    if pretrained:
        model = _load_pretrained(model, 'resnext50_32x4d')
    return model


def resnext101_32x8d(pretrained=False):
    model = ResNet(
        Bottleneck, [3, 4, 23, 3],
        groups=32, width_per_group=8
    )
    if pretrained:
        model = _load_pretrained(model, 'resnext101_32x8d')
    return model