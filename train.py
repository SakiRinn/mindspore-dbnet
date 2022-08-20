import yaml
import numpy as np

import mindspore
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore import context
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

from datasets.load import DataLoader
import modules.loss as loss
from modules.model import DBnet, DBnetPP, WithLossCell
from utils.callback import CkptSaver, LrScheduler, StepMonitor


def learning_rate_function(lr, cur_epoch_num, config):
    total_epochs = config['train']['total_epochs']
    start_epoch_num = config['train']['start_epoch_num']
    lr = config['optimizer']['lr']['value']
    factor = config['optimizer']['lr']['factor']

    rate = (1.0 - (start_epoch_num + cur_epoch_num) / (total_epochs + 1))**factor
    lr = rate * lr
    return lr


def train(path=None):
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    if config['train']['start_epoch_num'] >= config['train']['total_epochs']:
        print('Training cancelled due to invalid config.')
        return

    ## Dataset
    data_loader = DataLoader(config, isTrain=True)
    train_dataset = ds.GeneratorDataset(data_loader,
                                        ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
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
        print("Pretrained weight has been loaded.")
        mindspore.load_param_into_net(net, model_dict)

    ## Train
    config_ck = CheckpointConfig(save_checkpoint_steps=config['train']['save_steps'],
                                 keep_checkpoint_max=config['train']['max_checkpoints'])
    if config['train']['is_eval_before_saving']:
        ckpoint = CkptSaver(config, prefix=(config['net']),
                            directory=config['train']['output_dir'],
                            config=config_ck)
    else:
        ckpoint = ModelCheckpoint(prefix=(config['net']),
                                  directory=config['train']['output_dir'],
                                  config=config_ck)
    logfile = config['train']['output_dir'] + config['train']['log_filename'] + '.log'
    model.train(config['train']['total_epochs'] - config['train']['start_epoch_num'],
                train_dataset, dataset_sink_mode=False,
                callbacks=[StepMonitor(logfile), LrScheduler(learning_rate_function, config), ckpoint])

    # Need to stop at a certain time
    # config_ck = CheckpointConfig(keep_checkpoint_max=10)
    # ckpoint = ModelCheckpoint(prefix="DBnet", directory="./checkpoints/DBnetPP/", config=config_ck)
    # stop = StopCallBack(stop_epoch=2, stop_step=230)
    # model.train(config['train']['n_epoch'], train_dataset, dataset_sink_mode=False,
    #             callbacks=[LossMonitor(), LearningRateScheduler(learning_rate_function), ckpoint, stop])


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=3)
    train('checkpoints/1E650.ckpt')
    print("Train has completed.")
