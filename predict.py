import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import mindspore as ms
from mindspore import Tensor
from mindspore.dataset.vision.py_transforms import RandomColorAdjust, ToTensor, Normalize
from modules.model import DBnet, DBnetPP


class Predict():
    def __init__(self, net, param_dic):
        self.net = net
        param_dict = param_dic
        ms.load_param_into_net(self.net, param_dict)
        # ms.save_checkpoint(self.net, 'checkpoints/pthTOckpt/mmocrTOckpt_new.ckpt')
        
    def show(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (1280, 736))
        cv2.imwrite("./law.jpg", img)
        img = Image.fromarray(img)
        img = img.convert('RGB')
        
        img = ToTensor()(img)
        law_image = img
        img = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        img = np.expand_dims(img, axis=0)
        pred = self.net(ms.Tensor(img))
        mask = pred['binary'].asnumpy()
        mask = np.where(mask>0.7, 1, 0)
        output = (law_image*mask)[0]*255
        # output = Image.fromarray(output)
        # cv2.imshow('img', output)
        output=output.swapaxes(0,2)
        output=output.swapaxes(0,1)

        cv2.imwrite("./output.jpg", output)
        print("保存成功")
    
        
if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=7)
    ckpt = 'checkpoints/pthTOckpt/LiaoResnet18_epo_0_TOckpt.ckpt'
    param_dic = ms.load_checkpoint(ckpt)
    Pred = Predict(DBnet(isTrain=False), param_dic)
    Pred.show(img_path='data/test_images/img_1.jpg')