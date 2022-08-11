import sys
import time
from turtle import end_fill
import numpy as np

from mindspore import Tensor, context, nn, ops
import mindspore as ms
from mindspore import Tensor, nn, ops, context

# Input:
#             pred: A dict which contains predictions.
#                 thresh: The threshold prediction
#                 binary: The text segmentation prediction.
#                 thresh_binary: Value produced by `step_function(binary - thresh)`.
#             batch:
#                 gt: Text regions bitmap gt.
#                 mask: Ignore mask,
#                     pexels where value is 1 indicates no contribution to loss.
#                 thresh_mask: Mask indicates regions cared by thresh supervision.
#                 thresh_map: Threshold gt.

class L1BalanceCELoss(nn.Cell):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5):
        super(L1BalanceCELoss, self).__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred, gt, gt_mask, thresh_map, thresh_mask):

        # bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask)
        bce_loss_output = 0

        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)
            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss_output * self.bce_scale
        else:
            loss = bce_loss_output

        return loss


class DiceLoss(nn.Cell):

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def construct(self, pred: Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''

        if weights is not None:
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        return loss


class MaskL1Loss(nn.Cell):

    def __init__(self, eps=1e-6):

        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def construct(self, pred, gt, mask):

        mask_sum = mask.sum()

        loss = ((pred[:, 0] - gt).abs() * mask).sum() / (mask_sum + self.eps)

        return loss


class BalanceCrossEntropyLoss(nn.Cell):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):

        super(BalanceCrossEntropyLoss, self).__init__()

        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bceloss = nn.BCELoss()
        self.topk = ops.TopK(sorted=True)
        self.K = 100
        self.min = ops.Minimum()

        self.mask_len = 100
        self.dummy_negative_count = Tensor(15, dtype=ms.int32)
        self.mask = Tensor(np.arange(self.mask_len).astype(np.int32))
        self.sort = ops.Sort(descending=True)
        self.cast = ops.Cast()

    def construct(self, pred, gt, mask):

        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''

        # see this example for workaround of hard negative mining:
        # https://gitee.com/zhao_ting_v/ssd_benchmark/blob/master/src/ssd_benchmark.py

        pos = (gt * mask)
        neg = (mask - pos)

        positive_count = pos.sum()
        negative_count = neg.sum()

        negative_count = self.min(negative_count, positive_count * self.negative_ratio).astype(ms.int32)

        # loss = self.bceloss(pred, gt)
        loss = self.bceloss(pred, gt)[:, 0, :, :]

        positive_loss = (loss * pos)
        negative_loss = (loss * neg).view(-1) # (100,)

        # negative_loss, _ = self.sort(negative_loss) # sort the losses in descending order.

        negative_loss, _ = self.topk(negative_loss, len(negative_loss))

        min_neg_score = ops.gather(negative_loss, negative_count, 0) # minimum score of the topk loss

        masked_neg_loss = self.cast(negative_loss > min_neg_score, ms.int32) # filter out losses less than topk loss.

        ops.stop_gradient(masked_neg_loss)

        masked_neg_loss = masked_neg_loss * negative_loss

        balance_loss = (positive_loss.sum() + masked_neg_loss.sum())/(positive_count + negative_count + self.eps)

        return balance_loss
        # negative_loss, _ = self.topk(negative_loss, negative_count)
        # negative_loss = negative_loss[:negative_count]
        # balance_loss = (positive_loss.sum() + negative_loss.sum())/(positive_count + negative_count + self.eps)

        # return balance_loss


def test_old():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=5)

    pred = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/pred.npy"), dtype=ms.float32)
    gt = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/gt.npy"), dtype=ms.float32)
    gt_mask = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/mask.npy"), dtype=ms.float32)

    print(pred.shape, gt.shape, gt_mask.shape)

    thresh_map = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/thresh_maps.npy"), dtype=ms.float32)
    thresh_mask = Tensor(np.load("/opt/nvme1n1/wz/dbnet_torch/thresh_masks.npy"), dtype=ms.float32)

    # diceloss = DiceLoss()
    # print(diceloss.construct(pred,gt,gt_mask,None))

    # maskl1loss = MaskL1Loss()
    # print(maskl1loss.construct(pred,gt,mask))

    balance_loss = BalanceCrossEntropyLoss()
    print(balance_loss.construct(pred, gt, gt_mask))

    # pred_dict = {}
    # pred_dict['binary'] = pred
    # pred_dict['thresh'] = pred

    # l1balanceloss = L1BalanceCELoss()
    # print(l1balanceloss.construct(pred_dict, gt, gt_mask, thresh_map, thresh_mask))
    print("")


def test_new():

    pred = np.load("/old/wlh/DBnetpp_mindspore/test_np/pred.npy")
    pred = Tensor(pred)
    gt = np.load("/old/wlh/DBnetpp_mindspore/test_np/gt.npy")
    gt = Tensor(gt) # (1,640,640)
    mask = np.load("/old/wlh/DBnetpp_mindspore/test_np/mask.npy")
    mask = Tensor(mask) # (1,640,640)

    shrink_maps = pred[:, 0, :, :] # (1,640,640)
    threshold_maps = pred[:, 1, :, :]
    binary_maps = pred[:, 2, :, :]

    # pos = (gt * mask)
    # neg = (mask - pos)
    # positive_count = pos.sum().astype(ms.int32)
    # negative_count = neg.sum().astype(ms.int32)

    # negative_count = min(negative_count, (positive_count * 3.0).astype(ms.int32))

    # bceloss test
    start = time.time()
    bceloss = BalanceCrossEntropyLoss()

    # shrink_maps_small = Tensor(np.random.rand(1,10,10), dtype=ms.float32)
    # gt_small = Tensor(np.random.rand(1,10,10), dtype=ms.float32)
    # mask_small = Tensor(np.random.rand(1,10,10), dtype=ms.float32)
    # loss = bceloss(shrink_maps_small, gt_small, mask_small)
    # print(loss)

    # loss = bceloss(shrink_maps, gt, mask)
    # print(loss)
    # end = time.time()
    # print("time:{}".format(end-start))


    # MaskL1Loss test
    MaskL1 = MaskL1Loss()
    loss = MaskL1(threshold_maps, gt, mask)
    print(loss)

    # # DiceLoss test
    # Dice = DiceLoss()
    # loss = Dice(binary_maps, gt, mask)
    # print(loss)

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=3)
    test_new()