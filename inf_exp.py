import numpy as np
import yaml
import mindspore
from mindspore.train.model import Model
from mindspore import context

from modules.model import DBnet

if __name__=="__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=5)
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    net = eval(config['net'])(config, isTrain=False)

    path = "checkpoints/imagenet_PretrainedCkpt/gpu_weight_r18.ckpt"
    model_dict = mindspore.load_checkpoint(path)
    # print(model_dict)

    print(mindspore.load_param_into_net(net, model_dict))

    print("====================== weight(part) ============================")
    print(net.backbone.conv1.weight.data.asnumpy().reshape((-1,))[:10])

    np.random.seed(1)
    x = np.random.randn(1, 3, 640, 640)
    print("====================== input ============================")
    print(x)
    x = mindspore.Tensor(x, mindspore.float32)
    print(x)
    
    output = net(x)
    print("====================== output ===========================")
    print(output)