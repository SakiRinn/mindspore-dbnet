import sys
import time

import numpy as np
from mindspore import Tensor, nn, ops
import mindspore as ms

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

class L1BalanceCELoss(nn.LossBase):
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

        bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask)
        # bce_loss_output = 0

        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)
            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss_output * self.bce_scale
        else:
            loss = bce_loss_output

        return loss


class DiceLoss(nn.LossBase):

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


class MaskL1Loss(nn.LossBase):

    def __init__(self, eps=1e-6):

        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def construct(self, pred, gt, mask):

        mask_sum = mask.sum()

        loss = ((pred[:, 0] - gt).abs() * mask).sum() / (mask_sum + self.eps)

        return loss


class BalanceCrossEntropyLoss(nn.LossBase):
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

        loss = self.bceloss(pred, gt)
        # loss = self.bceloss(pred[:, 0, :, :], gt)

        positive_loss = (loss * pos)
        negative_loss = (loss * neg).view(-1) # (100,)

        negative_loss, _ = self.sort(negative_loss) # sort the losses in descending order.

        min_neg_score = ops.gather(negative_loss, negative_count, 0) # minimum score of the topk loss

        masked_neg_loss = self.cast(negative_loss > min_neg_score, ms.int32) # filter out losses less than topk loss.

        ops.stop_gradient(masked_neg_loss)

        masked_neg_loss = masked_neg_loss * negative_loss

        balance_loss = (positive_loss.sum() + masked_neg_loss.sum())/(positive_count + negative_count + self.eps)

        return balance_loss

def test_bce():
    pred = np.load("/old/wlh/DBnetpp_mindspore/test_np/pred.npy")
    pred = Tensor(pred)
    mask = np.load("/old/wlh/DBnetpp_mindspore/test_np/mask.npy")
    mask = Tensor(mask)
    gt = np.load("/old/wlh/DBnetpp_mindspore/test_np/gt.npy")
    gt = Tensor(gt)

    BCE = BalanceCrossEntropyLoss()

    # print(pred.shape, mask.shape, gt.shape, pred.dtype, mask.dtype, gt.dtype)
    # sys.exit()

    SHAPE = 640

    for _ in range(20):

        pred_random = Tensor(np.random.rand(1,3,SHAPE,SHAPE), dtype=ms.float32)
        gt_random = Tensor(np.random.rand(1,SHAPE,SHAPE), dtype=ms.float32)
        mask_random = Tensor(np.random.rand(1,SHAPE,SHAPE), dtype=ms.float32)

        start = time.time()
        loss = BCE(pred_random, gt_random, mask_random)
        end = time.time()

        print("loss ",loss)
        print("time ", end-start)


if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_id=5)
    test_bce()

# update sort():
# "/usr/local/Ascend/ascend-toolkit/5.0.4/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/sort.py"