import numpy as np
from mindspore import Tensor, export, load_checkpoint, load_param_into_net,context
import yaml
from datasets.load import DataLoader
import mindspore.dataset as ds
from modules.model import DBnet, DBnetPP



if __name__ == '__main__':
    

    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=6)


    net = eval(config['net'])(config, isTrain=False)

    path = 'checkpoints/dice/CurrentBest.ckpt'
    # load the parameter into net
    model_dict = load_checkpoint(path)
    load_param_into_net(net, model_dict)

    # data_loader = DataLoader(config, isTrain=False)
    # val_dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    # val_dataset = val_dataset.batch(1)
    # dataset = val_dataset.create_dict_iterator()

    input = np.random.uniform(0.0, 1.0, size=[1, 3, 736, 1280]).astype(np.float32)
    export(net, Tensor(input), file_name='DBnet_R18', file_format='MINDIR')