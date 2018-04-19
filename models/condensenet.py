from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from layers import Conv, LearnedGroupConv

__all__ = ['CondenseNet']

class BatchNormRelu(nn.Sequential):
    def __init__(self, in_channels):
        super(BatchNormRelu, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))

class DWConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super(DWConv, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(in_channels))
        # self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv-d', nn.Conv2d(in_channels, in_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding, bias=False,
                                            groups=in_channels))
        #  self.add_module('norm', nn.BatchNorm2d(in_channels))
        #  self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv-p', nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0, bias=False))

class GroupConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1):
        super(GroupConv, self).__init__()
        # self.add_module('norm', nn.BatchNorm2d(in_channels))
        # self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))

class _DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()

        cur_channels = in_channels
        c1_out_channels = args.bottleneck * growth_rate
        c2_out_channels = growth_rate

        self.b1 = BatchNormRelu(in_channels)
        self.c1 = GroupConv(in_channels, c1_out_channels, kernel_size=3, padding=1, groups=args.num_groups)
        self.b2 = BatchNormRelu(c1_out_channels)
        self.c2 = DWConv(c1_out_channels, c2_out_channels, kernel_size=3, padding=1, groups=args.num_groups)
        
    def forward(self, x):
        x_ = x
        x = self.b1(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        return torch.cat([x_, x], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, args):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(in_channels + i * growth_rate, growth_rate, args)
            self.add_module('denselayer_%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, in_channels, args):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNet(nn.Module):
    def __init__(self, args):

        super(CondenseNet, self).__init__()

        self.stages = args.stages
        self.growth = args.growth
        assert len(self.stages) == len(self.growth)
        self.args = args
        self.progress = 0.0
        if args.data in ['cifar10', 'cifar100']:
            self.init_stride = 1
            self.pool_size = 8
        else:
            self.init_stride = 2
            self.pool_size = 7

        self.features = nn.Sequential()
        ### Initial nChannels should be 3
        self.num_features = 2 * self.growth[0]
        ### Dense-block 1 (224x224)
        self.features.add_module('init_conv', nn.Conv2d(3, self.num_features,
                                                        kernel_size=3,
                                                        stride=self.init_stride,
                                                        padding=1,
                                                        bias=False))
        for i in range(len(self.stages)):
            ### Dense-block i
            self.add_block(i)
        ### Linear layer
        self.classifier = nn.Linear(self.num_features, args.num_classes)

        ### initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        return

    def add_block(self, i):
        ### Check if ith is the last one
        last = (i == len(self.stages) - 1)
        block = _DenseBlock(
            num_layers=self.stages[i],
            in_channels=self.num_features,
            growth_rate=self.growth[i],
            args=self.args,
        )
        self.features.add_module('denseblock_%d' % (i + 1), block)
        self.num_features += self.stages[i] * self.growth[i]
        if not last:
            trans = _Transition(in_channels=self.num_features,
                                args=self.args)
            self.features.add_module('transition_%d' % (i + 1), trans)
        else:
            self.features.add_module('norm_last',
                                     nn.BatchNorm2d(self.num_features))
            self.features.add_module('relu_last',
                                     nn.ReLU(inplace=True))
            self.features.add_module('pool_last',
                                     nn.AvgPool2d(self.pool_size))

    def forward(self, x, progress=None):
        if progress:
            LearnedGroupConv.global_progress = progress
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out
