import yaml
import numpy as np
import time
import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore.train.model import Model
from mindspore import context
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint

from datasets.load import DataLoader
import modules.loss as loss
from modules.model import DBnet, DBnetPP, WithLossCell
from utils.callback import CkptSaver, LrScheduler, StepMonitor
from eval import WithEvalCell
from eval import evaluate


def learning_rate_function(lr, cur_epoch_num, config):
    total_epochs = config['train']['total_epochs']
    start_epoch_num = config['train']['start_epoch_num']
    lr = config['optimizer']['lr']['value']
    factor = config['optimizer']['lr']['factor']

    rate = (1.0 - (start_epoch_num + cur_epoch_num) / (total_epochs + 1))**factor
    lr = rate * lr
    return lr

def train_epoch(epoch, net, dataset, config):
    net.set_train(True)
    # dataset_size = dataset.get_dataset_size()
    losses = []
    for i, data in enumerate(dataset):
        loss = net(*data)
        if isinstance(loss, tuple):
            if loss[1]:
                print("============ overflow! ============", flush=True)
            loss = loss[0]
        loss = loss.asnumpy()
        losses.append(loss)
        cur_lr = net.optimizer.learning_rate.data.asnumpy()
        loss_log = "[%s] epoch: %d step: %2d lr: %.6f, loss is %.6f" % \
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                    epoch, i, cur_lr, np.mean(losses))
        print(loss_log, flush=True)
    new_lr = learning_rate_function(net.optimizer.learning_rate.data.asnumpy(), epoch, config)
    ms.ops.assign(net.optimizer.learning_rate, ms.Tensor(new_lr, ms.float32))


def eval_epoch(epoch, train_net, eval_net, dataset, config, max_f):
    ms.save_checkpoint(train_net, f"cur_epoch.ckpt")
    ms.load_checkpoint("cur_epoch.ckpt", eval_net.model)

    eval_net.model.set_train(False)
    metrics, _ = eval_net.eval(dataset, show_imgs=config['eval']['show_images'])

    cur_f = metrics['fmeasure'].avg
    print(f"\ncurrent epoch is {epoch}, current fmeasure is {cur_f}")
    if cur_f >= max_f:
        print(f"update best ckpt at epoch {epoch}, best fmeasure is {cur_f}\n")
        ms.save_checkpoint(eval_net.model, f"best_epoch.ckpt")
        max_f = cur_f
    return max_f


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
    train_dataset = train_dataset.batch(config['train']['batch_size'], drop_remainder=True)
    train_dataset = train_dataset.create_tuple_iterator()

    val_data_loader = DataLoader(config, isTrain=False)
    val_dataset = ds.GeneratorDataset(val_data_loader, ['original', 'img', 'polys', 'dontcare'])
    val_dataset = val_dataset.batch(1)
    val_dataset = val_dataset.create_dict_iterator()

    ## Model & Loss & Optimizer
    net = eval(config['net'])(config, isTrain=True)
    optim = nn.SGD(params=net.trainable_params(),
                   learning_rate=config['optimizer']['lr']['value'],
                   momentum=config['optimizer']['momentum'],
                   weight_decay=config['optimizer']['weight_decay'])
    criterion = loss.L1BalanceCELoss(**config['loss'])
    net_with_loss = WithLossCell(net.to_float(ms.float16), criterion.to_float(ms.float32))
    train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss,
                                                 optimizer=optim,
                                                 scale_sense=nn.FixedLossScaleUpdateCell(1024.))
    max_f = 0
    eval_net = eval(config['net'])(config, isTrain=False)
    eval_net = WithEvalCell(eval_net, config)
    for epoch in range(config['train']['start_epoch_num'], config['train']['total_epochs']):
        train_epoch(epoch, train_net, train_dataset, config)
        if epoch!=0 and epoch%10 == 0:
            max_f = eval_epoch(epoch, train_net, eval_net, val_dataset, config, max_f)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=6)
    train()
    print("Train has completed.")
