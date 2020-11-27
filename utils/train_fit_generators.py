"""
该脚本仿照keras的训练时信息打印方式, 集成后方便调用
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.nn_metrics import get_auc_score, get_accuracy_score, get_f1_score, get_precision_score, get_recall_score
from utils.nn_metrics import get_average_precision_score


class TrainFitGenerator(object):
    """
    模型保存的回调函数

    Args:
    net (torch.nn.Moudle): 神经网络
    optimizer(torch.optim): 优化器
    loss_function: 损失函数
    generator: 数据加载器
    epochs(int): 训练轮数, 默认是1
    validation_data: 验证集的数据加载器
    callbacks: 回调函数
    accumulate: 梯度累计次数
    average： 计算recall、 precision、f1时的方法, binary为二分类时使用
    """
    def __init__(self, net, optimizer, loss_function, generator, epochs=1,
                 validation_data=None, callbacks=None, metrics=None,
                 accumulate=1):
        self.net = net
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.generator = generator
        self.epochs = epochs
        self.steps_per_epoch = len(self.generator)
        self.validation_data = validation_data
        self.validation_steps = len(self.validation_data) if self.validation_data is not None else None
        self.accumulate = accumulate
        self. metrics = metrics
        self.history = {"loss": [], "acc": []}
        self.callbacks = callbacks
        self.init_callbacks()

        if self.metrics is not None:
            for k in self.metrics:
                self.history[k] = []

        if self.validation_data is not None:
            history_keys = list(self.history.keys())
            for k in history_keys:
                self.history["val_" + k] = []

    def init_callbacks(self):
        if self.callbacks is not None:
            for callback_function in self.callbacks:
                callback_function.set_model(self.net, self.optimizer)

    def epoch_end_run_callbacks(self, epoch):
        if self.callbacks is not None:
            for callback_function in self.callbacks:
                callback_function.on_epoch_end(epoch, self.history)

    def epoch_begin_run_callbacks(self, epoch):
        if self.callbacks is not None:
            for callback_function in self.callbacks:
                callback_function.on_epoch_begin(epoch, self.history)

    def run(self, start_epoch=0):
        self.optimizer.zero_grad()
        for epoch in range(start_epoch, self.epochs):
            batch_train_metrics = dict([(k, []) for k in self.history.keys() if "val_" not in k])
            self.net.train()
            self.epoch_begin_run_callbacks(epoch)
            with tqdm(self.generator, desc='Train Epoch[{}/{}]'.format(epoch+1, self.epochs),
                      file=sys.stdout, disable=False) as iterator:
                for step, (data, target) in enumerate(iterator):
                    ni = step + epoch * self.steps_per_epoch
                    data = data.cuda()
                    target = target.cuda()
                    output = self.net(data)
                    loss = self.loss_function(output, target)
                    loss.backward()

                    # 是否进行梯度累计
                    if ni % self.accumulate == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    batch_train_metrics["loss"].append(loss.item())
                    y_true = target.cpu().numpy()
                    y_pred_prob = F.softmax(output.detach().cpu(), dim=-1).numpy()
                    y_pred = np.argmax(y_pred_prob, axis=-1)
                    acc = get_accuracy_score(y_true=y_true, y_pred=y_pred)
                    batch_train_metrics["acc"].append(acc)
                    if self.metrics is not None:
                        for k, func in self.metrics.items():
                            batch_train_metrics[k].append(func(y_true, y_pred))
                    str_logs = ['{}: {:.4}'.format(k, v[-1]) for k, v in batch_train_metrics.items()]
                    s = " - ".join(str_logs)
                    iterator.set_postfix_str(s)

                # 保存历史指标
                for key in batch_train_metrics.keys():
                    self.history[key].append(np.mean(batch_train_metrics[key]))
                str_logs = ['{}: {:.4}'.format(k, v[-1]) for k, v in self.history.items() if "val" not in k]
                s = " - ".join(str_logs)
                iterator.set_postfix_str(s)

            if self.validation_data is not None:
                self.net.eval()
                step_val_loss = []
                step_y_true = []
                step_y_positive_prob = []
                step_y_pred = []
                step_val_metrics = dict([(k, []) for k in self.history.keys() if "val_" in k])
                with torch.no_grad():
                    with tqdm(self.validation_data, desc='Validation Epoch [{}/{}]'.format(epoch+1, self.epochs),
                              file=sys.stdout, disable=False) as val_iterator:
                        for i, (data, target) in enumerate(val_iterator):
                            data = data.cuda()
                            target = target.cuda()
                            output = self.net(data)
                            val_loss = self.loss_function(output, target)
                            step_val_loss.append(val_loss.item())
                            step_y_true.extend(target.cpu().numpy().tolist())
                            step_y_positive_prob.extend(F.softmax(output.detach().cpu(), dim=-1).numpy()[:, -1].tolist())
                            step_y_pred.extend(np.argmax(output.detach().cpu().numpy(), axis=-1).tolist())

                            if i == self.validation_steps-1:
                                step_val_metrics["val_loss"] = np.mean(step_val_loss)
                                val_acc = get_accuracy_score(y_true=step_y_true, y_pred=step_y_pred)
                                step_val_metrics["val_acc"] = val_acc
                                if self.metrics is not None:
                                    for k, func in self.metrics.items():
                                        step_val_metrics["val_"+k] = func(y_true, y_pred)
                                str_logs = ['{}: {:.4}'.format(k, v) for k, v in step_val_metrics.items()]
                                s = " - ".join(str_logs)
                                val_iterator.set_postfix_str(s)

                    for key in step_val_metrics.keys():
                        self.history[key].append(step_val_metrics[key])

            self.epoch_end_run_callbacks(epoch)