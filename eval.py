import os
import time
import numpy as np
import yaml
import cv2
from tqdm.auto import tqdm

import mindspore
from mindspore import Tensor, context
import mindspore.dataset as ds
import mindspore.nn as nn

import sys
sys.path.insert(0, '.')
from datasets.load import DataLoader
from utils.metric import QuadMetric
from utils.post_process import SegDetectorRepresenter
from modules.model import DBnet, DBnetPP


class WithEvalCell(nn.Cell):
    def __init__(self, model, config):
        super(WithEvalCell, self).__init__(auto_prefix=False)
        self.model = model
        self.config = config
        self.metric = QuadMetric(config['eval']['polygon'])
        self.post_process = SegDetectorRepresenter(config['eval']['thresh'], config['eval']['box_thresh'],
                                                   config['eval']['max_candidates'],
                                                   config['eval']['unclip_ratio'],
                                                   config['eval']['polygon'],
                                                   config['eval']['dest'])

    def construct(self, batch):
        start = time.time()

        preds = self.model(batch['img'])
        boxes, scores = self.post_process(preds)
        raw_metric = self.metric.validate_measure(batch, (boxes, scores))

        cur_frame = batch['img'].shape[0]
        cur_time = time.time() - start

        return raw_metric, (cur_frame, cur_time)

    def eval(self, dataset, show_imgs=True):
        total_frame = 0.0
        total_time = 0.0
        raw_metrics = []
        count = 0

        for batch in tqdm(dataset):
            raw_metric, (cur_frame, cur_time) = self(batch)
            raw_metrics.append(raw_metric)

            print('\n', raw_metric['evaluationLog'], end='')
            print(f"Recall: {raw_metric['recall']}, Precision: {raw_metric['precision']}, Hmean: {raw_metric['hmean']}")
            total_frame += cur_frame
            total_time += cur_time

            count += 1
            if show_imgs:
                img = batch['original'].asnumpy().squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gtPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gtDontCare']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4)
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4)
                # pred
                for idx, poly in enumerate(raw_metric['detPolys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['detDontCare']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4)
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                if not os.path.exists('images'):
                    os.makedirs('images')
                cv2.imwrite(f'images/eval_{count}.jpg', img)

        metrics = self.metric.gather_measure(raw_metrics)
        print(f'FPS: {total_frame / total_time}')
        print(metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg)

def evaluate(path: str):
    ## Config
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()

    ## Dataset
    data_loader = DataLoader(config, isTrain=False)
    val_dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    val_dataset = val_dataset.batch(1)
    dataset = val_dataset.create_dict_iterator()

    ## Model
    net = eval(config['net'])(config, isTrain=False)
    model_dict = mindspore.load_checkpoint(path)
    mindspore.load_param_into_net(net, model_dict)

    ## Eval
    eval_net = WithEvalCell(net, config)
    eval_net.set_train(False)
    eval_net.eval(dataset)


if __name__ == '__main__':
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=6)
    evaluate('./checkpoints/DBnet/DBnet_2-29_27.ckpt')
