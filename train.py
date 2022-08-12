import yaml
import numpy as np

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import LearningRateScheduler, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.model import Model
from mindspore import context, load_checkpoint

from datasets.load import DataLoader
import modules.loss as loss
from modules.model import DBnet, DBnetPP, WithLossCell, StopCallBack


def learning_rate_function(lr, cur_epoch_num):
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    epochs = config['train']['epochs']
    lr = config['optimizer']['lr']['value']
    factor = config['optimizer']['lr']['factor']

    rate = np.power(1.0 - cur_epoch_num / float(epochs + 1), factor)
    return rate * lr


def train(path=None):
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, isTrain=True)
    train_dataset = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                                        num_parallel_workers=config['dataset']['num_workers'])
    train_dataset = train_dataset.batch(config['train']['batch_size'])

    ## Model & Loss & Optimizer
    net = eval(config['net'])(config, isTrain=True)
    optim = nn.SGD(params=net.trainable_params(),
                   learning_rate=config['optimizer']['lr']['value'],
                   momentum=config['optimizer']['momentum'],
                   weight_decay=config['optimizer']['weight_decay'])
    criterion = loss.L1BalanceCELoss(**config['loss'])
    net_with_loss = WithLossCell(net, criterion)
    model = Model(net_with_loss, optimizer=optim)

    ## Resume
    if path is not None:
        model_dict = mindspore.load_checkpoint(path)
        mindspore.load_param_into_net(net, model_dict)

    ## Train
    config_ck = CheckpointConfig(save_checkpoint_steps=config['train']['save_steps'],
                                 keep_checkpoint_max=config['train']['max_checkpoints'])
    ckpoint = ModelCheckpoint(prefix=eval(config['net']),
                              directory=config['train']['output_dir'],
                              config=config_ck)
    model.train(config['train']['epochs'], train_dataset, dataset_sink_mode=False,
                callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint])

    #need to stop at a certain time
    # config_ck = CheckpointConfig(keep_checkpoint_max=10)
    # ckpoint = ModelCheckpoint(prefix="DBnet", directory="./checkpoints/DBnetPP/", config=config_ck)
    # stop = StopCallBack(stop_epoch=2, stop_step=230)
    # model.train(config['train']['n_epoch'], train_dataset, dataset_sink_mode=False,
    #             callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint, stop])


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)
    train()
    print("Train has completed.")
