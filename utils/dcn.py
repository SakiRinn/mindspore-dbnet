"""
Deformable Convolution operator V2
"""
import os
import numpy as np

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops as ops

np.random.seed(0)
ms.common.set_seed(0)


@ops.constexpr
def _get_offset_base(offset_shape, stride):
    """
    get base position index from deformable shift of each kernel element.
    """
    # (n, 2*k*k, h, w)
    k2, h, w = offset_shape[1] // 2, offset_shape[2], offset_shape[3]
    k = int(k2**0.5)
    # (k)
    range_pn = np.arange(-(k - 1) // 2, (k - 1) // 2 + 1)
    # (k, k), (k, k)
    p_n_x, p_n_y = np.meshgrid(range_pn, range_pn)
    # (k*k, 1), (k*k, 1) -> (k*k, 2)
    p_n = np.concatenate((p_n_x.reshape(k2, 1), p_n_y.reshape(k2, 1)), axis=0)
    # (k*k, 2) -> (1, 2*k*k, 1, 1)
    p_n = p_n.reshape(1, 2 * k2, 1, 1)

    # (h)
    # range_h = nn.Range(1, h * self.stride + 1, self.stride)()
    # (w)
    # range_w = nn.Range(1, w * self.stride + 1, self.stride)()
    range_h = np.arange(k // 2, h * stride + 1, stride)
    range_w = np.arange(k // 2, w * stride + 1, stride)
    # (h, w), (h, w)
    p_0_x, p_0_y = np.meshgrid(range_h, range_w, indexing='xy')

    # (h, w) -> (1, 1, h, w)
    p_0_x = p_0_x.reshape(1, 1, h, w)
    # (1, 1, h, w) -> (1, k*k, h, w)
    p_0_x = np.tile(p_0_x, (1, k2, 1, 1))

    # (h, w) -> (1, 1, h, w)
    p_0_y = p_0_y.reshape(1, 1, h, w)
    # (1, 1, h, w) -> (1, k*k, h, w)
    p_0_y = np.tile(p_0_y, (1, k2, 1, 1))

    # (1, k*k, h, w), (1, k*k, h, w) -> (1, 2*k*k, h, w)
    p_0 = np.concatenate((p_0_x, p_0_y), axis=1)
    # (1, 2*k*k, h, w) + (1, 2*k*k, 1, 1) + (n, 2*k*k, h, w) -> (n, 2*k*k, h, w)
    p = p_0 + p_n
    return ms.Tensor(p.astype(np.float32))


def _get_feature_by_index(x, p_h, p_w):
      """gather feature by specified index"""
      # x (n, c, h_in, w_in)
      # p_h (n, h, w, k*k)
      # p_w (n, h, w, k*k)
      n, c, h_in, w_in = x.shape
      _, h, w, k2 = p_h.shape
      # (n, c, h_in, w_in) -> (n, h_in, w_in, c)
      x = x.transpose(0, 2, 3, 1)

      # the following is the opt for:
      # input(n, h_in, w_in, c), index_x/index_y(n, h, w, k*k) -> output(n, h, w, k*k, c)

      # (n, h_in, w_in, c) -> (n*h_in*w_in, c)
      x = x.reshape(-1, c)

      # (n)
      idx_0_n = nn.Range(0, n, 1)()
      # (n, h, w, k*k) + (n, h, w, k*k) + (n, 1, 1, 1)
      index = p_w + p_h * w_in + idx_0_n.reshape(n, 1, 1, 1) * w_in * h_in

      # (n*h_in*w_in, c), (n, h, w, k*k) -> (n, h, w, k*k, c)
      x_offset = ops.Gather()(x, index, 0)
      # (n, h*w*k*k, c) -> (n, h*w, k*k, c)
      x_offset = x_offset.reshape(n, h * w, k2, c)
      # (n, h*w, k*k, c) -> (n, c, h*w, k*k)
      x_offset = x_offset.transpose(0, 3, 1, 2)
      # (n, c, h*w, k*k) -> (n, c, h, w, k*k)
      x_offset = x_offset.reshape(n, c, h, w, k2)
      return x_offset


def _regenerate_feature_map(x_offset):
    """ get rescaled feature map which was enlarged by ks**2 times."""
    # offset (n, c, h, w, k*k)
    n, c, h, w, k2 = x_offset.shape
    k = ops.ScalarCast()(k2 ** 0.5, mstype.int32)
    # issue??
    # (n, c, h, w, k*k) -> k * (n, c, h, w, k)
    splits = ops.Split(axis=-1, output_num=k)(x_offset)
    # k * (n, c, h, w, k) -> (n, c, k*h, w, k)
    x_offset = ops.Concat(axis=2)(splits)
    # (n, c, k*h, w, k) -> (n, c, h*k, w*k)
    x_offset = x_offset.reshape(n, c, h * k, w * k)
    return x_offset


class DeformConv2d(nn.Cell):
    """
    Deformable convolution opertor

    Args:
        inc(int): Input channel.
        outc(int): Output channel.
        kernel_size (int): Convolution window. Default: 3.
        stride (int): The distance of kernel moving. Default: 1.
        padding (int): Implicit paddings size on both sides of the input. Default: 1.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: False.
        modulation (bool): If True, modulated defomable convolution (Deformable ConvNets v2). Default: True.
    Returns:
        Tensor, detection of images(bboxes, score, keypoints and category id of each objects)
    """
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, has_bias=False, modulation=True):
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, pad_mode='valid', padding=0,
                              stride=kernel_size, has_bias=has_bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=self.kernel_size,
                                pad_mode='pad', padding=self.padding, stride=self.stride)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=self.kernel_size,
                                    pad_mode='valid', padding=0, stride=self.stride)
        if kernel_size % 2 == 0:
            raise ValueError("Only odd number is supported, but current kernel sizeis {}".format(kernel_size))

    def construct(self, x):
        """deformed sampling locations with augmented offsets"""
        # 0 ── h ──x
        # |
        # w
        # |
        # y

        # (n, c, h_in, w_in)
        x_shape = x.shape
        # get learned shift for each pixels(the shift is relative to current pixel)
        # (n, c, h_in, w_in) -> (n, 2*k*k, h, w)
        offset = self.p_conv(x)
        if self.padding > 0:
            # (n, c, h_in+2p, w_in+2p)
            x = self.zero_padding(x)

        # get absolute postion of each pixel w.r.s to input feature map without offset
        # -> (1, 2*k*k, h, w)
        p_base = _get_offset_base(offset.shape, self.stride)

        p = p_base + offset

        # (n, 2*k*k, h, w) -> (n, h, w, 2*k*k)
        p = p.transpose(0, 2, 3, 1)
        p_lt = ops.Floor()(p).astype(mstype.int32)
        p_rb = p_lt + 1

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        k2 = p.shape[-1] // 2
        p_h = p[:,:,:, :k2].clip(0, x_shape[2] - 1)
        p_w = p[:,:,:, k2:].clip(0, x_shape[3] - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_lt_h = p_lt[:,:,:, :k2].clip(0, x_shape[2] - 1)
        p_lt_w = p_lt[:,:,:, k2:].clip(0, x_shape[3] - 1)

        # (n, h, w, 2*k*k) -> (n, h, w, k*k), (n, h, w, k*k)
        p_rb_h = p_rb[:,:,:, :k2].clip(0, x_shape[2] - 1)
        p_rb_w = p_rb[:,:,:, k2:].clip(0, x_shape[3] - 1)

        # perform bilinear interpolation
        # (n, h, w, k*k) -> (n, h, w, k*k)
        # issue??
        weight_lt = (1 + (p_lt_h - p_h)) * (1 + (p_lt_w - p_w))
        weight_rb = (1 - (p_rb_h - p_h)) * (1 - (p_rb_w - p_w))
        weight_lb = (1 + (p_lt_h - p_h)) * (1 - (p_rb_w - p_w))
        weight_rt = (1 - (p_rb_h - p_h)) * (1 + (p_lt_w - p_w))

        # (n, c, h_in, w_in), (n, h, w, k*k), (n, h, w, k*k) -> (n, c, h, w, k*k)
        x_p_lt = _get_feature_by_index(x, p_lt_h, p_lt_w)
        x_p_rb = _get_feature_by_index(x, p_rb_h, p_rb_w)
        x_p_lb = _get_feature_by_index(x, p_lt_h, p_rb_w)
        x_p_rt = _get_feature_by_index(x, p_rb_h, p_lt_w)

        # (n, h, w, k*k) -> (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
        x_offset = (ops.ExpandDims()(weight_lt, 1) * x_p_lt +
                    ops.ExpandDims()(weight_rb, 1) * x_p_rb +
                    ops.ExpandDims()(weight_lb, 1) * x_p_lb +
                    ops.ExpandDims()(weight_rt, 1) * x_p_rt)

        if self.modulation:
            # modulation (b, 1, h, w, N)
            # (n, c, h, w) -> (n, k*k, h, w)
            m = ops.Sigmoid()(self.m_conv(x))
            # (n, k*k, h, w) -> (n, h, w, k*k)
            m = m.transpose(0, 2, 3, 1)
            # (n, h, w, k*k) -> (n, 1, h, w, k*k)
            m = ops.ExpandDims()(m, 1)
            # (n, 1, h, w, k*k) * (n, c, h, w, k*k) -> (n, c, h, w, k*k)
            x_offset = x_offset * m
        # (n, c, h, w, k*k) -> (n, c, h*k, w*k)
        x_offset = _regenerate_feature_map(x_offset)
        # (n, c, h*k, w*k) -> (n, c, h, w)
        out = self.conv(x_offset)
        return out

# test
if __name__ == '__main__':
    ms.context.set_context(mode=ms.context.GRAPH_MODE, save_graphs=True, save_graphs_path='irs')
    from mindspore.profiler import Profiler
    profiler  = Profiler()
    x = ms.Tensor(np.random.randint(0, 255, (2, 64, 128, 128)), ms.float32)
    dcn_x = DeformConv2d(64, 8, 3, 1)
    b = dcn_x(x)
    import time
    start = time.time()
    for i in range(100):
        b = dcn_x(x).asnumpy()
    end = time.time()
    profiler.analyse()
    print("--"*20)
    print("cal time:", (end-start)/100)
    print("--"*20)
    print(b.shape)
    if os.path.exists('benchmark.npy'):
        benchmark = np.load('benchmark.npy')
        assert(np.allclose(b, benchmark, rtol=1e-5))
        print('test is passed!')
    else:
        np.save('benchmark.npy', b)
        print('benchmark is saved!')
