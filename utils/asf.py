import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common import initializer as init


class ScaleFeatureSelection(nn.Cell):
    def __init__(self, in_channels, inter_channels , out_features_num=4,
                 attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels=in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num

        self.conv = nn.Conv2d(in_channels, inter_channels, 3, has_bias=True)
        self.type = attention_type
        if self.type == 'scale_spatial':
            self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4,
                                                            out_features_num)
        elif self.type == 'scale_channel_spatial':
            self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 4,
                                                                   out_features_num)
        elif self.type == 'scale_channel':
            self.enhanced_attention = ScaleChannelAttention(inter_channels, inter_channels//2,
                                                            out_features_num)
        else:
            exit(1)
        self.interpolate = nn.ResizeBilinear()

    def weights_init(self, c):
        for m in c.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = init.initializer(init.HeNormal(), m.weight.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def construct(self, concat_x, features_list):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)

        # assert len(features_list) == self.out_features_num
        if len(features_list) != self.out_features_num:
            exit(1)
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = self.interpolate(score, shape)

        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])

        return ops.Concat(axis=1)(x)


class ScaleChannelAttention(nn.Cell):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelAttention, self).__init__()
        self.avgpool = ops.ReduceMean(keep_dims=True)
        self.fc1 = nn.Conv2d(in_planes, out_planes, 1, has_bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.fc2 = nn.Conv2d(out_planes, num_features, 1, has_bias=False)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(axis=1)
        if init_weight:
            self.weights_init()

    def weights_init(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                            m.weight.shape)
                if m.bias is not None:
                    m.bias = init.initializer('zeros', m.bias.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def construct(self, x):
        global_x = self.avgpool(x, (-2, -1))
        global_x = self.fc1(global_x)
        global_x = self.relu(self.bn(global_x))
        global_x = self.fc2(global_x)
        global_x = self.softmax(global_x)
        return global_x


class ScaleChannelSpatialAttention(nn.Cell):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.SequentialCell(
            nn.Conv2d(in_planes, out_planes , 1, has_bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, has_bias=False)
        )
        self.spatial_wise = nn.SequentialCell(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, has_bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.SequentialCell(
            nn.Conv2d(in_planes, num_features, 1, has_bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        if init_weight:
            self.weights_init()

    def weights_init(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                            m.weight.shape)
                if m.bias is not None:
                    m.bias = init.initializer('zeros', m.bias.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def construct(self, x):
        # global_x = self.avgpool(x)
        #shape Nx4x1x1
        x = ops.ReduceMean(keep_dims=True)(x, (-2, -1))
        global_x = self.sigmoid(self.channel_wise(x))
        #shape: NxCxHxW
        global_x = global_x + x
        #shape:Nx1xHxW
        x = ops.ReduceMean(keep_dims=True)(global_x, 1)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleSpatialAttention(nn.Cell):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.SequentialCell(
            #Nx1xHxW
            nn.Conv2d(1, 1, 3, has_bias=False),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, has_bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.SequentialCell(
            nn.Conv2d(in_planes, num_features, 1, has_bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self.weights_init()

    def weights_init(self):
        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = init.initializer(init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                            m.weight.shape)
                if m.bias is not None:
                    m.bias = init.initializer('zeros', m.bias.shape)
            elif isinstance(m, nn.BatchNorm2d):
                m.gamma = init.initializer('ones', m.gamma.shape)
                m.beta = init.initializer(1e-4, m.beta.shape)

    def construct(self, x):
        global_x = ops.ReduceMean(keep_dims=True)(x, 1)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


if __name__ == '__main__':
    ...