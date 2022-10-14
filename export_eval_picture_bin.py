from datasets.load import DataLoader
import mindspore
from mindspore import context
import yaml
import numpy as np
import os


if __name__ == '__main__':
    stream = open('config.yaml', 'r', encoding='utf-8')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    stream.close()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=0)

    import mindspore.dataset as ds
    data_loader = DataLoader(config, isTrain=False)
    dataset = ds.GeneratorDataset(data_loader, ['original', 'img', 'polys', 'dontcare'])
    dataset = dataset.batch(1)
    it = dataset.create_dict_iterator(output_numpy=True)

    data_path = "./eval_pic_bin"
    os.makedirs(data_path)

    for i,data in enumerate(it):
        file_name = "eval_pic_" + str(i+1) + ".bin"
        file_path = os.path.join(data_path, file_name)
        data['img'].tofile(file_path)
        print(data['img'].shape)

    print("finished")