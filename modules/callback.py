import math
import time
import numpy as np

from mindspore import log as logger
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train.callback import Callback
from utils.metric import AverageMeter


class StopCallBack(Callback):
    def __init__(self, stop_epoch, stop_step):
        super(StopCallBack,self).__init__()
        self.stop_step = stop_step
        self.stop_epoch = stop_epoch

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        if epoch_num == self.stop_epoch and cur_step_in_epoch==self.stop_step:
            run_context.request_stop()


class StepMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        per_print_times (int): How many steps to print once loss. During sink mode, it will print loss in the
                               nearest step. Default: 1.

    Raises:
        ValueError: If per_print_times is not an integer or less than zero.
    """

    def __init__(self, logname, per_print_times=1):
        super(StepMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0

        self.loss_avg = AverageMeter()
        self.logname = logname

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if cb_params.net_outputs is not None:
            loss = cb_params.net_outputs.asnumpy()
        else:
            print("custom loss callback class loss is None.")
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        cur_lr = cb_params.optimizer.learning_rate.asnumpy().item()

        if cur_step_in_epoch == 1:
            self.loss_avg = AverageMeter()
        self.loss_avg.update(loss)

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            loss_log = "[%s] epoch: %d step: %2d lr: %.15f, loss is %.6f" % \
                       (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
                       cb_params.cur_epoch_num, cur_step_in_epoch, cur_lr, np.mean(self.loss_avg.avg))
            print(loss_log, flush=True)
            with open(self.logname, "a+") as loss_file:
                loss_file.write(loss_log)
                loss_file.write("\n")


class LrScheduler(Callback):
    """
    Change the learning_rate during training.

    Args:
        learning_rate_function (Function): The function about how to change the learning rate during training.

    Examples:
        >>> from mindspore import Model
        >>> from mindspore.train.callback import LearningRateScheduler
        >>> import mindspore.nn as nn
        ...
        >>> def learning_rate_function(lr, cur_step_num):
        ...     if cur_step_num%1000 == 0:
        ...         lr = lr*0.1
        ...     return lr
        ...
        >>> lr = 0.1
        >>> momentum = 0.9
        >>> net = Net()
        >>> loss = nn.SoftmaxCrossEntropyWithLogits()
        >>> optim = nn.Momentum(net.trainable_params(), learning_rate=lr, momentum=momentum)
        >>> model = Model(net, loss_fn=loss, optimizer=optim)
        ...
        >>> dataset = create_custom_dataset("custom_dataset_path")
        >>> model.train(1, dataset, callbacks=[LearningRateScheduler(learning_rate_function)],
        ...             dataset_sink_mode=False)
    """

    def __init__(self, learning_rate_function):
        super(LrScheduler, self).__init__()
        self.learning_rate_function = learning_rate_function

    def step_end(self, run_context):
        """
        Change the learning_rate at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        arr_lr = cb_params.optimizer.learning_rate.asnumpy()
        lr = float(np.array2string(arr_lr))
        new_lr = self.learning_rate_function(lr, cb_params.cur_epoch_num)
        if not math.isclose(lr, new_lr, rel_tol=1e-10):
            F.assign(cb_params.optimizer.learning_rate, Tensor(new_lr, mstype.float32))
            logger.info(f'At step {cb_params.cur_epoch_num}, learning_rate change to {new_lr}')