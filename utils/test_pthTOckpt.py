from collections import OrderedDict
import torch
import sys
sys.path.append(".")
# from modules.model import DBnet, DBnetPP
import mindspore as ms
import numpy
import yaml

def mod_Liao(param_dic):
    new_dic = OrderedDict()
    for key, value in param_dic.items():

        if 'conv' in key and 'layer' not in key:
            key = key[-12:]

        if 'bn' in key and 'weight' in key and 'layer' not in key:
            key = key[-10:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' not in key:
            key = key[-8:-4] + 'beta'

        if 'conv' in key and 'layer' in key:
            key = key[-21:]
        if 'bn' in key and 'weight' in key and 'layer' in key:
            key = key[-19:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' in key:
            key = key[-17:-4] + 'beta'

        if 'downsample.0' in key and 'weight' in key:
            key = key[-28:]
        if 'downsample.1' in key and 'weight' in key:
            key = key[-28:-6] + 'gamma'
        if 'downsample.1' in key and 'bias' in key:
            key = key[-26:-4] + 'beta'

        if 'fc' in key and 'weight' in key:
            key = key[-9:]
        if 'fc' in key and 'bias' in key:
            key = key[-7:]

        if 'in' in key and 'bina' not in key and 'run' not in key:
            key = key[-10:]

        if 'out2' in key:
            key = 'out2.weight'
        if 'out3' in key:
            key = 'out3.weight'
        if 'out4' in key:
            key = 'out4.weight'
        if 'out5' in key:
            key = 'out5.weight'

        if ('binarize.0' in key or 'binarize.3' in key or 'binarize.6' in key) and ('weight' in key):
            key = key[-17:]
        if 'binarize.3.bias' in key or 'binarize.6.bias' in key:
            key = key[-15:]
        if 'binarize.1.weight' in key or 'binarize.4.weight' in key:
            key = key[-17:-6] + 'gamma'
        if 'binarize.1.bias' in key or 'binarize.4.bias' in key:
            key = key[-15:-4] + 'beta'

        if ('thresh.0' in key or 'thresh.3' in key or 'thresh.6' in key) and ('weight' in key):
            key = key[-15:]
        if 'thresh.3.bias' in key or 'thresh.6.bias' in key:
            key = key[-13:]
        if 'thresh.1.weight' in key or 'thresh.4.weight' in key:
            key = key[-15:-6] + 'gamma'
        if 'thresh.1.bias' in key or 'thresh.4.bias' in key:
            key = key[-13:-4] + 'beta'

        if 'bn' in key and 'running_mean' in key and 'layer' not in key:
            key = 'bn1.moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' not in key:
            key = 'bn1.moving_variance'

        if 'bn' in key and 'running_mean' in key and 'layer' in key:
            key = key[-25:-12] + 'moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' in key:
            key = key[-24:-11] + 'moving_variance'

        if 'downsample' in key and 'running_mean' in key:
            key = key[-34:-12] + 'moving_mean'
        if 'downsample' in key and 'running_var' in key:
            key = key[-33:-11] + 'moving_variance'

        if 'binarize' in key and 'running_mean' in key:
            key = key[-23:-12] + 'moving_mean'
        if 'binarize' in key and 'running_var' in key:
            key = key[-22:-11] + 'moving_variance'

        if 'thresh' in key and 'running_mean' in key:
            key = key[-21:-12] + 'moving_mean'
        if 'thresh' in key and 'running_var' in key:
            key = key[-20:-11] + 'moving_variance'

        new_dic[key] = ms.Parameter(ms.Tensor(value.numpy()).astype(ms.float32))
    return new_dic


def mod_mmocr(param_dic):
    new_dic = OrderedDict()
    param_dic = param_dic['state_dict']
    for key, value in param_dic.items():

        if 'conv' in key and 'layer' not in key and 'neck' not in key:
            key = key[-12:]

        if 'bn' in key and 'weight' in key and 'layer' not in key:
            key = key[-10:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' not in key:
            key = key[-8:-4] + 'beta'

        if 'conv' in key and 'layer' in key:
            key = key[-21:]
        if 'bn' in key and 'weight' in key and 'layer' in key:
            key = key[-19:-6] + 'gamma'
        if 'bn' in key and 'bias' in key and 'layer' in key:
            key = key[-17:-4] + 'beta'

        if 'downsample.0' in key and 'weight' in key:
            key = key[-28:]
        if 'downsample.1' in key and 'weight' in key:
            key = key[-28:-6] + 'gamma'
        if 'downsample.1' in key and 'bias' in key:
            key = key[-26:-4] + 'beta'

        if 'fc' in key and 'weight' in key:
            key = key[-9:]
        if 'fc' in key and 'bias' in key:
            key = key[-7:]

        if 'neck.lateral_convs.0.conv.weight' in key:
            key = 'in2.weight'
        if 'neck.lateral_convs.1.conv.weight' in key:
            key = 'in3.weight'
        if 'neck.lateral_convs.2.conv.weight' in key:
            key = 'in4.weight'
        if 'neck.lateral_convs.3.conv.weight' in key:
            key = 'in5.weight'

        # if 'neck.smooth_convs.0.conv.weight' in key:
        #     key = 'out2.weight'
        # if 'neck.smooth_convs.1.conv.weight' in key:
        #     key = 'out3.weight'
        # if 'neck.smooth_convs.2.conv.weight' in key:
        #     key = 'out4.weight'
        # if 'neck.smooth_convs.3.conv.weight' in key:
        #     key = 'out5.weight'

        if 'neck.smooth_convs.0.conv.weight' in key:
            key = 'out5.weight'
        if 'neck.smooth_convs.1.conv.weight' in key:
            key = 'out4.weight'
        if 'neck.smooth_convs.2.conv.weight' in key:
            key = 'out3.weight'
        if 'neck.smooth_convs.3.conv.weight' in key:
            key = 'out2.weight'

        if ('binarize.0' in key or 'binarize.3' in key or 'binarize.6' in key) and ('weight' in key):
            key = key[-17:]
        if 'binarize.3.bias' in key or 'binarize.6.bias' in key:
            key = key[-15:]
        if 'binarize.1.weight' in key or 'binarize.4.weight' in key:
            key = key[-17:-6] + 'gamma'
        if 'binarize.1.bias' in key or 'binarize.4.bias' in key:
            key = key[-15:-4] + 'beta'

        if ('threshold.0' in key or 'threshold.3' in key or 'threshold.6' in key) and ('weight' in key):
            key = 'thresh' + key[-9:]
        if 'threshold.3.bias' in key or 'threshold.6.bias' in key:
            key = 'thresh' + key[-7:]
        if 'threshold.1.weight' in key or 'threshold.4.weight' in key:
            key = 'thresh' + key[-9:-6] + 'gamma'
        if 'threshold.1.bias' in key or 'threshold.4.bias' in key:
            key = 'thresh' + key[-7:-4] + 'beta'

        if 'bn' in key and 'running_mean' in key and 'layer' not in key:
            key = 'bn1.moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' not in key:
            key = 'bn1.moving_variance'

        if 'bn' in key and 'running_mean' in key and 'layer' in key:
            key = key[-25:-12] + 'moving_mean'
        if 'bn' in key and 'running_var' in key and 'layer' in key:
            key = key[-24:-11] + 'moving_variance'

        if 'downsample' in key and 'running_mean' in key:
            key = key[-34:-12] + 'moving_mean'
        if 'downsample' in key and 'running_var' in key:
            key = key[-33:-11] + 'moving_variance'

        if 'binarize' in key and 'running_mean' in key:
            key = key[-23:-12] + 'moving_mean'
        if 'binarize' in key and 'running_var' in key:
            key = key[-22:-11] + 'moving_variance'

        if 'threshold' in key and 'running_mean' in key:
            key = 'thresh' + key[-15:-12] + 'moving_mean'
        if 'threshold' in key and 'running_var' in key:
            key = 'thresh' + key[-14:-11] + 'moving_variance'

        new_dic[key] = ms.Parameter(ms.Tensor(value.numpy()).astype(ms.float32))
    return new_dic


def mod_pretrained_resnet(param_dic):
    new_dic = OrderedDict()
    for key,value in param_dic.items():
        if 'bn1d' in key:
            key = key[:9] + 'bn1' + key[13:]
        if 'bn2d' in key:
            key = key[:9] + 'bn2' + key[13:]
        if 'down_sample_layer' in key:
            key = key[:9] + 'downsample' + key[26:]

        new_dic[key] = ms.Parameter(value)

    return new_dic



if __name__ == "__main__":
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=3)

    ## load pth
    # param_dic = torch.load("tests/pre-trained-model-synthtext-resnet18", map_location=torch.device('cpu'))
    # new_dic = mod_Liao(param_dic=param_dic)

    # Pred = Predict(DBnet(isTrain=False), new_dic)
    # Pred.show(img_path='data/train_images/img_3.jpg')
    # print("完成")stream = open('config.yaml', 'r', encoding='utf-8')

    # load ckpt
    param_dic = ms.load_checkpoint("checkpoints/imagenet_PretrainedCkpt/resnet18_Weight.ckpt")
    new_dic = mod_pretrained_resnet(param_dic)


    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    net = DBnet(config)
    ms.load_param_into_net(net, new_dic)
    ms.save_checkpoint(net, 'checkpoints/imagenet_PretrainedCkpt/pretrained_R18_DB.ckpt')