# coding:utf-8
from utils.dcn import DeformConv2d

import mindspore as ms
from mindspore import ops, Tensor
from mindspore import context, set_seed
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal, Constant
from mindspore import load_checkpoint, load_param_into_net

import numpy as np
import math
from collections import OrderedDict


# set initializer to constant for debugging.
def conv3x3(inplanes, outplanes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, pad_mode="pad",
                     padding=1, weight_init="ones")


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()

        # print("BasicBlock out_channels", inplanes, planes)

        # set initializer to constant for debugging.
        self.with_dcn = dcn is not None

        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=None, momentum=0.1)

        self.relu = nn.ReLU()

        if self.with_dcn:

            deformable_groups = dcn.get('deformable_groups', 1)

            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1, stride=1)

        else:
            self.conv2 = conv3x3(planes, planes)

        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=None, momentum=0.1)

        self.downsample = downsample

        self.stride = stride

    def construct(self, x):
        residual = x
        # print("进入block")
        # print(x.shape)

        out = self.conv1(x)
        # print("conv1 ",out[0][0][0][:5])

        out = self.bn1(out)
        # print("bn1 ",out[0][0][0][:5])

        out = self.relu(out)
        # print("relu1 ",out[0][0][0][:5])

        out = self.conv2(out)
        # print("conv2 ",out[0][0][0][:5])

        out = self.bn2(out)
        # print("bn2 ",out[0][0][0][:5])

        if self.downsample is not None:
            residual = self.downsample(x)
        # print(out.shape)
        # print(residual.shape)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        # print("define bottleneck")
        self.with_dcn = dcn is not None

        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, has_bias=False, weight_init="ones")

        # self.bn1 = BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=None, momentum=0.1)

        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            self.conv2 = DeformConv2d(planes, planes, kernel_size=3, padding=1)

        else:
            # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, has_bias=False, pad_mode="pad",
                                   padding=1, weight_init="ones")

        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=None, momentum=0.1)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, has_bias=False, weight_init="ones")

        # self.bn3 = BatchNorm2d(planes * 4)
        self.bn3 = nn.BatchNorm2d(planes * 4, use_batch_statistics=None, momentum=0.1)

        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def construct(self, x):
        # print("进入bottleneck")
        residual = x
        # print("x.shape:{}".format(x.shape))
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


class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, dcn=None):

        self.inplanes = 64

        super(ResNet, self).__init__()

        # TODO: set initializer to constant for debugging.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, pad_mode="pad",
                               has_bias=False, weight_init="ones")  # same卷积，图片尺寸不变

        self.bn1 = nn.BatchNorm2d(64, use_batch_statistics=None, momentum=0.1)
        self.relu = nn.ReLU()

        # TODO: pytorch maxpool2d sets padding=1 but mindspore maxpool2d can't. so just use pad_mode.
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dcn=dcn)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        for m in self.cells():

            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_init = Normal(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):

        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample的功能是调整residual使其和out保持相同尺寸，out的变化由plane和stride控制
            downsample = nn.SequentialCell(
                # set initializer to constant for debugging.
                nn.Conv2d(self.inplanes, planes * block.expansion, pad_mode="pad",
                          kernel_size=1, stride=stride, has_bias=False, weight_init="ones"),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=None, momentum=0.1),
            )

        layers = []

        layers.append(block(self.inplanes, planes,
                            stride, downsample, dcn))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dcn=dcn))

        return nn.SequentialCell(*layers)

    def construct(self, x):

        # print("x ",x[0][0][0][:10])

        x = self.conv1(x)

        # print("x conv1 ", x[0][0][0][:5])
        # print("conv1 WEIGHTS ", self.conv1.weight)

        x = self.bn1(x)

        # print("x bn1 ",x[0][0][0][:5])

        x = self.relu(x)

        x = self.maxpool(x)
        # print("x maxpool",x[0][0][0][:5])

        x2 = self.layer1(x)
        # print("x2 ",x2[0][0][0][:5])
        # print(x2.shape)
        x3 = self.layer2(x2)
        # print("x3 ",x3[0][0][0][:5])

        x4 = self.layer3(x3)
        # print("x4 ",x4[0][0][0][:5])

        x5 = self.layer4(x4)
        # print("x5 ",x5[0][0][0][:5])

        return x2, x3, x4, x5


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        ms_dict = load_checkpoint("/opt/nvme1n1/wz/dbnet_torch/path-to-model-directory/res18_ms.ckpt")
        param_not_load = load_param_into_net(model, ms_dict)

    return model


def deformable_resnet18(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)

    if pretrained:
        ms_dict = load_checkpoint("")
        param_not_load = load_param_into_net(model, ms_dict)

    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    ms_dict = load_checkpoint("")
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        param_not_load = load_param_into_net(model, ms_dict)

    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    ms_dict = load_checkpoint("")
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)

    if pretrained:
        param_not_load = load_param_into_net(model, ms_dict)

    return model


def test_conv():
    ones = ops.Ones()

    data = ones((1, 64, 184, 320), ms.float32)

    print("原尺寸：{}".format(data.shape))

    conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, pad_mode="pad",
                     padding=1, weight_init="ones")

    output = conv(data)

    print("卷积后尺寸：{}".format(output.shape))

    print(output[0][0][1][:100])


def test_dcn():
    ones = ops.Ones()

    data = ones((1, 64, 184, 320), ms.float32)

    print("原尺寸：{}".format(data.shape))

    conv = DeformConv2d(64, 64, kernel_size=3, padding=1, stride=1)

    output = conv(data)

    print("卷积后尺寸：{}".format(output.shape))

    print(output[0][0][1][:100])


def test_bn():
    bn = nn.BatchNorm2d(64, momentum=0.1, use_batch_statistics=None)

    data = np.load("/opt/nvme1n1/wz/dbnet_torch/bninput.npy")

    output = bn(Tensor(data, dtype=ms.float32))

    print(output[0][0][0][:5])


def test_basicblock():
    block = BasicBlock(inplanes=64, planes=64)

    # np.random.seed(0)
    # data = np.random.rand(1,64,184,320)

    ones = ops.Ones()

    data = ones((1, 64, 184, 320), ms.float32)

    # print("test BasicBlock input ", data[0][0][0][:100])

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = block(inp_tensor)

    print(output.shape)
    print("test BasicBlock output ", output[0][3][3][:100])


def test_Bottleneck():
    block = Bottleneck(inplanes=64, planes=64)

    # np.random.seed(0)
    # data = np.random.rand(1,64,184,320)

    ones = ops.Ones()

    data = ones((1, 64, 256, 256), ms.float32)

    # print("test BasicBlock input ", data[0][0][0][:100])

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = block(inp_tensor)

    print(output.shape)
    print("test Bottleneck output ", output[0][3][3][:100])


def test_deformable_resnet18():
    data = np.load("/old/wlh/DBnetpp_mindspore/dbnet/test.npy")

    print("原图大小为：{}".format(data.shape))
    resnet = ResNet(BasicBlock, [2, 2, 2, 2], dcn={'deformable_groups': 1})

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = resnet(inp_tensor)

    for t in output:
        print(t.shape)


def test_resnet18():
    data = np.load("/old/wlh/DBnetpp_mindspore/dbnet/test.npy")

    print("原图大小为：{}".format(data.shape))
    resnet = ResNet(BasicBlock, [2, 2, 2, 2])

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = resnet(inp_tensor)

    for t in output:
        print(t.shape)
    print(output[0][0][0][1][:100])


def test_resnet50():
    data = np.load("/old/wlh/DBnetpp_mindspore/dbnet/test.npy")

    print("原图大小为：{}".format(data.shape))
    resnet = ResNet(Bottleneck, [3, 4, 6, 3])

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = resnet(inp_tensor)

    for t in output:
        print(t.shape)
    print(output[0][0][0][1][:100])


def test_deformative_resnet50():
    data = np.load("/old/wlh/DBnetpp_mindspore/dbnet/test.npy")

    print("原图大小为：{}".format(data.shape))
    resnet = ResNet(Bottleneck, [3, 4, 6, 3], dcn={'deformable_groups': 1})

    inp_tensor = Tensor(data, dtype=ms.float32)

    output = resnet(inp_tensor)

    for t in output:
        print(t.shape)
    print(output[0][0][0][1][:100])


if __name__ == "__main__":
    context.set_context(device_id=2, mode=context.GRAPH_MODE)
    test_resnet50()
