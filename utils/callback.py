"""
该脚本实现为train_fit_generators调用的回调函数
"""
import numpy as np
import torch
import csv


class Callback(object):
    """
    回调函数的模板类
    """
    def __init__(self):
        self.model = None
        self.optimizer = None

    def set_model(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ModelCheckpoint(Callback):
    """
    模型保存的回调函数

    Args:
    filepath (string): 权重文件保存路径
    monitor(str): 监控指标，用于作为保存权重文件的根据
    verbose(int): 是否打印输出信息
    save_best_only(bool): 是否仅保存最佳权重, 默认是False
    mode(str): 监控指标的比较方式, 默认是min
    save_weights_only(bool): 是否仅保存权重文件, 如果为True连模型结构一起保存, 默认是False
    period(int): 保存的轮数, 默认是1
    """
    def __init__(self, filepath, monitor="val_acc", verbose=1,
                 save_best_only=False, mode="max", save_weights_only=False,
                 period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode not in ['min', 'max']:
            raise ValueError('ModelCheckpoint mode %s is unknown, '
                             'fallback to auto mode.' % (mode))
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)[-1]
                if current is None:
                    raise ValueError('Can save best model only with %s available, skipping.' % (self.monitor))
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s\n'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            torch.save({"state_dict": self.model.state_dict(),
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "epoch": epoch + 1},
                                       filepath)
                        else:
                            torch.save({"model": self.model,
                                        "optimizer_state_dict": self.optimizer.state_dict(),
                                        "epoch": epoch + 1},
                                       filepath)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve from %0.5f\n' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s\n' % (epoch + 1, filepath))
                if self.save_weights_only:
                    torch.save({"state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "epoch": epoch + 1},
                               filepath)
                else:
                    torch.save({"model": self.model,
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "epoch": epoch + 1},
                               filepath)


class SaveEpochMetrics(Callback):
    """
    模型指标保存的回调函数

    Args:
    filepath (string): 指标文件保存路径
    period(int): 保存的轮数, 默认是1
    """
    def __init__(self, filepath, period=1):
        super(SaveEpochMetrics, self).__init__()
        self.filepath = filepath
        self.period = period
        self.epochs_since_last_save = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            train_metrics = ["loss", "acc"]
            train_metrics.extend([k for k in logs.keys() if "val_" not in k and k not in ["loss", "acc"]])
            val_metrics = ["val_"+k for k in train_metrics] if "val_loss" in logs.keys() else []
            tile = ["epoch"] + train_metrics + val_metrics

            with open(self.filepath, 'w', encoding="utf-8-sig") as fp:
                fconv = csv.writer(fp)
                fconv.writerow(tile)

                for i in range(len(logs["loss"])):
                    result = [i+1]
                    for k in train_metrics+val_metrics:
                        result.append(round(logs[k][i], 4))
                    fconv.writerow(result)


class ExpLrScheduler(Callback):
    """
    模型学习率衰减的回调函数

    Args:
    init_lr (float): 初始学习率, 默认是0.01
    lr_decay_epoch(int): 衰减的间隔轮数, 默认是10
    verbose(int): 是否打印输出信息
    weight_decay(float): 衰减比例, 默认是0.8
    """
    def __init__(self, init_lr=0.01, lr_decay_epoch=10, weight_decay=0.8):
        super(ExpLrScheduler, self).__init__()
        if init_lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(init_lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.weight_decay = weight_decay

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.init_lr * (self.weight_decay ** (epoch // self.lr_decay_epoch))
        print('LR is set to {}'.format(lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr