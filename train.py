import yaml
import numpy as np

import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.callback import LearningRateScheduler, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.train.model import Model
from mindspore import context, load_checkpoint

from datasets.load import DataLoader
import modules.loss as loss
from modules.model import DBnet, DBnetPP, WithLossCell, StopCallBack


def learning_rate_function(lr, cur_epoch_num):
    lr = 0.007
    epochs = 1200
    factor = 0.9

    rate = np.power(1.0 - cur_epoch_num / float(epochs + 1), factor)

    return rate * lr


def train():
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, isTrain=True)
    train_dataset = ds.GeneratorDataset(data_loader, ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'])
    train_dataset = train_dataset.batch(config['train']['batch_size'])
    # default batch size 16. dataset size 63.

    ## Model & Loss & Optimizer
    net = DBnet(isTrain=True)
    optim = nn.SGD(params=net.trainable_params(), learning_rate=0.007, momentum=0.9, weight_decay=1e-4)
    criterion = loss.L1BalanceCELoss()
    net_with_loss = WithLossCell(net, criterion)
    model = Model(net_with_loss, optimizer=optim)

    ## Train
    config_ck = CheckpointConfig(keep_checkpoint_max=10)
    ckpoint = ModelCheckpoint(prefix="DBnet", directory="./checkpoints/DBnet/", config=config_ck)
    model.train(config['train']['n_epoch'], train_dataset, dataset_sink_mode=False,
                callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint])

    #need to stop at a certain time
    # config_ck = CheckpointConfig(keep_checkpoint_max=10)
    # ckpoint = ModelCheckpoint(prefix="DBnet", directory="./checkpoints/DBnetPP/", config=config_ck)
    # stop = StopCallBack(stop_epoch=2, stop_step=230)
    # model.train(config['train']['n_epoch'], train_dataset, dataset_sink_mode=False,
    #             callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint, stop])


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    train()
    print("Train has completed.")
