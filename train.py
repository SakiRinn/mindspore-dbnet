import os
import yaml
import numpy as np
import time
import argparse

import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size

from datasets.load import DataLoader
import modules.loss as loss
from modules.model import DBnet, DBnetPP, WithLossCell
from utils.callback import LrScheduler, StepMonitor, CkptSaver
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint


def learning_rate_function(cur_epoch_num, config):
    total_epochs = config['train']['total_epochs']
    start_epoch_num = config['train']['start_epoch_num']
    lr = config['optimizer']['lr']['value']
    factor = config['optimizer']['lr']['factor']

    rate = (1.0 - (start_epoch_num + cur_epoch_num) / (total_epochs + 1))**factor
    lr = rate * lr
    return lr

def init_env(cfg):
    ms.set_seed(cfg["seed"])

    if cfg["device_target"] != "None":
        if cfg["device_target"] not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg['device_target']}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg["device_target"])

    if cfg["context_mode"] not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg['context_mode']}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg["context_mode"] == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg["device_target"] = ms.get_context("device_target")
    if cfg["device_target"] == "CPU":
        cfg["device_id"] = 0
        cfg["device_num"] = 1
        cfg["rank_id"] = 0

    if hasattr(cfg, "device_id") and isinstance(cfg["device_id"], int):
        ms.set_context(device_id=cfg["device_id"])

    if cfg["device_num"] > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg["device_num"] != group_size:
            raise ValueError(f"the setting device_num: {cfg['device_num']} not equal to the real group_size: {group_size}")
        cfg["rank_id"] = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if "all_reduce_fusion_config" in cfg:
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg["all_reduce_fusion_config"])
    else:
        cfg["device_num"] = 1
        cfg["rank_id"] = 0
        print("run standalone!", flush=True)

def init_group_params(net, weight_decay):
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    return group_params

def train():
    ## Config
    parser = argparse.ArgumentParser(description="default name", add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config_path", type=str, default=os.path.join(current_dir, "config.yaml"),
                        help="Config file path")
    parser.add_argument("--device_num", type=int, default=1, help="Device numbers")
    path_args, _ = parser.parse_known_args()

    stream = open(path_args.config_path, 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    if config['train']['start_epoch_num'] >= config['train']['total_epochs']:
        print('Training cancelled due to invalid config.')
        return
    config["device_num"] = path_args.device_num
    init_env(config)    ## Dataset
    data_loader = DataLoader(config, isTrain=True)
    train_dataset = ds.GeneratorDataset(data_loader,
                                        ['img', 'gts', 'gt_masks', 'thresh_maps', 'thresh_masks'],
                                        num_parallel_workers=config['dataset']['num_workers'],
                                        num_shards=config["device_num"], shard_id=config["rank_id"],
                                        shuffle=True, max_rowsize=32)
    train_dataset = train_dataset.batch(config['train']['batch_size'], drop_remainder=True)

    ## Setup
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

    ## Model & Loss & Optimizer
    net = eval(config['net'])(config, isTrain=True)
    opt = nn.Momentum(params=init_group_params(net, config['optimizer']['weight_decay']),
                        learning_rate=config['optimizer']['lr']['value'],
                        momentum=config['optimizer']['momentum'])
    criterion = loss.L1BalanceCELoss(**config['loss'])
    net_with_loss = WithLossCell(net, criterion)
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=opt,
                                                 scale_sense=nn.FixedLossScaleUpdateCell(1024.))
    model = ms.Model(train_net)
    model.train(config['train']['total_epochs'], train_dataset,
                callbacks=[StepMonitor(logfile), LrScheduler(learning_rate_function, config), ckpoint])

if __name__ == '__main__':
    train()
    print("Train has completed.")
