import numpy as np

import mindspore as ms
from mindspore import ops, Tensor
from mindspore import context
import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
from mindspore.common import initializer as init

import sys
sys.path.insert(0, '.')
from utils.asf import ScaleFeatureSelection


class SegDetector(nn.Cell):
    def __init__(self, in_channels=[64, 128, 256, 512], inner_channels=256, k=50,
                 bias=False, adaptive=True, smooth=False, serial=False, training=False):
        '''
        in_channels:resnet18=[64, 128, 256, 512]
                    resnet50=[2048,1024,512,256]
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''

        super(SegDetector, self).__init__()

        self.k = k
        self.serial = serial
        self.training = training

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1, has_bias=bias, weight_init=HeNormal())
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1, has_bias=bias, weight_init=HeNormal())
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1, has_bias=bias, weight_init=HeNormal())
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1, has_bias=bias, weight_init=HeNormal())

        self.out5 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias,
                              weight_init=HeNormal())
        self.out4 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias,
                              weight_init=HeNormal())
        self.out3 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias,
                              weight_init=HeNormal())
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 4, 3, pad_mode="pad", padding=1, has_bias=bias,
                              weight_init=HeNormal())

        self.binarize = nn.SequentialCell(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, pad_mode="pad", padding=1, has_bias=bias, weight_init=HeNormal()),
            nn.BatchNorm2d(inner_channels // 4, use_batch_statistics=None, momentum=0.1),
            nn.ReLU(),
            nn.Conv2dTranspose(inner_channels // 4, inner_channels // 4, 2, stride=2, has_bias=True),
            nn.BatchNorm2d(inner_channels // 4, use_batch_statistics=None, momentum=0.1),
            nn.ReLU(),
            nn.Conv2dTranspose(inner_channels // 4, 1, 2, stride=2, has_bias=True),
            nn.Sigmoid())

        self.weights_init(self.binarize)

        self.adaptive = adaptive

        if adaptive:
            self.thresh = self._init_thresh(
                inner_channels, serial=serial, bias=bias)
            self.weights_init(self.thresh)

        self.weights_init(self.in5)
        self.weights_init(self.in4)
        self.weights_init(self.in3)
        self.weights_init(self.in2)

        self.weights_init(self.out5)
        self.weights_init(self.out4)
        self.weights_init(self.out3)
        self.weights_init(self.out2)

    def weights_init(self, c):

        for m in c.cells():

            if isinstance(m, nn.Conv2dTranspose):

                m.weight = init.initializer(HeNormal(), m.weight.shape)
                m.bias = init.initializer('zeros', m.bias.shape)

            elif isinstance(m, nn.Conv2d):

                m.weight = init.initializer(HeNormal(), m.weight.shape)

            elif isinstance(m, nn.BatchNorm2d):

                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def _init_thresh(self, inner_channels, serial=False, bias=False):

        in_channels = inner_channels

        if serial:
            in_channels += 1

        self.thresh = nn.SequentialCell(
            nn.Conv2d(in_channels, inner_channels // 4,
                      3, pad_mode="pad", padding=1, has_bias=bias, weight_init=HeNormal()),  # plane:1024->256
            nn.BatchNorm2d(inner_channels // 4, use_batch_statistics=None, momentum=0.1),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, in_channels // 4, 2, stride=2, weight_init=HeNormal(), has_bias=True),
            # size*2
            nn.BatchNorm2d(inner_channels // 4, use_batch_statistics=None, momentum=0.1),
            nn.ReLU(),
            nn.Conv2dTranspose(in_channels // 4, 1, 2, stride=2, weight_init=HeNormal(), has_bias=True),  # size*2, plane=1
            nn.Sigmoid())

        return self.thresh

    def construct(self, features):

        # shapes for inference:
        # torch.Size([1, 64, 184, 320])
        # torch.Size([1, 128, 92, 160])
        # torch.Size([1, 256, 46, 80])
        # torch.Size([1, 512, 23, 40])

        c2, c3, c4, c5 = features

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        # 进行上采样，准备进行连接操作
        up5 = ops.ResizeNearestNeighbor((in4.shape[2], in4.shape[3]))
        up4 = ops.ResizeNearestNeighbor((in3.shape[2], in3.shape[3]))
        up3 = ops.ResizeNearestNeighbor((in2.shape[2], in2.shape[3]))

        out4 = up5(in5) + in4  # 1/16
        out3 = up4(out4) + in3  # 1/8
        out2 = up3(out3) + in2  # 1/4

        upsample = ops.ResizeNearestNeighbor((c2.shape[2], c2.shape[3]))

        # 将连接后的结果再进行上采样，使其形状相同，1/4
        p5 = upsample(self.out5(in5))
        p4 = upsample(self.out4(out4))
        p3 = upsample(self.out3(out3))
        p2 = upsample(self.out2(out2))

        concat = ops.Concat(1)
        fuse = concat((p5, p4, p3, p2))  # size:1/4.plane:1024

        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)

        pred = {}
        pred['binary'] = binary

        if self.adaptive and self.training:
            thresh = self.thresh(fuse)

            pred['thresh'] = thresh
            pred['thresh_binary'] = self.step_function(binary, thresh)

        return pred

    def step_function(self, x, y):
        """
        通过binary, thresh得到二值化图
        """
        reciprocal = ops.Reciprocal()
        exp = ops.Exp()

        return reciprocal(1 + exp(-self.k * (x - y)))


class SegDetectorPP(SegDetector):
    def __init__(self, in_channels=[64, 128, 256, 512], inner_channels=256, k=10,
                 bias=False, adaptive=True, smooth=False, serial=False, training=False,
                 attention_type='scale_spatial'):
        super(SegDetectorPP, self).__init__(in_channels, inner_channels, k,
                                            bias, adaptive, smooth, serial, training)
        self.concat_attention = ScaleFeatureSelection(inner_channels, inner_channels//4,
                                                      attention_type=attention_type)
        self.concat_attention.weights_init(self.concat_attention)

    def construct(self, features):

        # shapes for inference:
        # torch.Size([1, 64, 184, 320])
        # torch.Size([1, 128, 92, 160])
        # torch.Size([1, 256, 46, 80])
        # torch.Size([1, 512, 23, 40])

        c2, c3, c4, c5 = features

        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        # 进行上采样，准备进行连接操作
        up5 = ops.ResizeNearestNeighbor((in4.shape[2], in4.shape[2]))
        up4 = ops.ResizeNearestNeighbor((in3.shape[2], in3.shape[2]))
        up3 = ops.ResizeNearestNeighbor((in2.shape[2], in2.shape[2]))

        out4 = up5(in5) + in4  # 1/16
        out3 = up4(out4) + in3  # 1/8
        out2 = up3(out3) + in2  # 1/4

        upsample = ops.ResizeNearestNeighbor((c2.shape[2], c2.shape[3]))

        # 将连接后的结果再进行上采样，使其形状相同，1/4
        p5 = upsample(self.out5(in5))
        p4 = upsample(self.out4(out4))
        p3 = upsample(self.out3(out3))
        p2 = upsample(self.out2(out2))

        # Different from DBNet
        fuse = ops.Concat(axis=1)((p5, p4, p3, p2))
        fuse = self.concat_attention(fuse, [p5, p4, p3, p2])

        # this is the pred module, not binarization module;
        # We do not correct the name due to the trained model.
        binary = self.binarize(fuse)

        pred = {}

        pred['binary'] = binary

        if self.adaptive and self.training:
            thresh = self.thresh(fuse)

            pred['thresh'] = thresh
            pred['thresh_binary'] = self.step_function(binary, thresh)

        return pred


if __name__ == "__main__":
    context.set_context(device_id=5, mode=context.GRAPH_MODE)

    print("segdetector ms test")

    c2 = np.random.rand(16, 64, 160, 160)
    c3 = np.random.rand(16, 128, 80, 80)
    c4 = np.random.rand(16, 256, 40, 40)
    c5 = np.random.rand(16, 512, 20, 20)

    c2 = Tensor(c2, dtype=ms.float32)
    c3 = Tensor(c3, dtype=ms.float32)
    c4 = Tensor(c4, dtype=ms.float32)
    c5 = Tensor(c5, dtype=ms.float32)

    segdetector = SegDetectorPP(adaptive=True, training=True, attention_type='scale_spatial')
    output = segdetector([c2, c3, c4, c5])

    print("segdetector output ", output[2].shape)   # 'thresh_binary'
