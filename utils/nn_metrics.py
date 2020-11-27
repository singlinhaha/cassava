"""
该脚本实现各种神经网络指标的计算
"""
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, average_precision_score
import numpy as np


def data_transform(data):
    """
    将数据转成数组
    """
    if isinstance(data, list):
        data = np.array(data)
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise ValueError("data style must be list or array")
    return data


def get_auc_score(y_true, y_score):
    """
    计算auc
    :param y_true: (list or numpy.array) 标签
    :param y_score: (list or numpy.array) 正样本概率
    :return:
    """
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    try:
        score = roc_auc_score(y_true=y_true, y_score=y_score)
    except:
        score = np.nan
    return score


def get_accuracy_score(y_true, y_pred):
    """
    计算准确率
    :param y_true: (list or numpy.array) 标签
    :param y_pred: (list or numpy.array) 预测值
    :return:
    """
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = accuracy_score(y_true=y_true, y_pred=y_pred)
    return score


def get_f1_score(y_true, y_pred, average='binary'):
    """
    计算非f1
    :param y_true: (list or numpy.array) 标签
    :param y_pred: (list or numpy.array) 预测值
     :average: 计算指标时的方法
    :return:
    """
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return score


def get_recall_score(y_true, y_pred, average='binary'):
    """
    计算召回率
    :param y_true: (list or numpy.array) 标签
    :param y_pred: (list or numpy.array) 预测值
     :average: 计算指标时的方法
    :return:
    """
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = recall_score(y_true=y_true, y_pred=y_pred, average=average)
    return score


def get_precision_score(y_true, y_pred, average='binary'):
    """
    计算精确率
    :param y_true: (list or numpy.array) 标签
    :param y_pred: (list or numpy.array) 预测值
    :average: 计算指标时的方法
    :return:
    """
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    score = precision_score(y_true=y_true, y_pred=y_pred, average=average)
    return score


def get_average_precision_score(y_true, y_score):
    """
    计算mAP
    :param y_true: (list or numpy.array) 标签
    :param y_score: (list or numpy.array) 正样本概率
    :return:
    """
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    score = average_precision_score(y_true=y_true, y_score=y_score)
    return score


def get_roc_best_threshold(y_true, y_score, pos_label=None):
    """
    计算roc曲线下最佳阈值
    :param y_true: (list or numpy.array) 标签
    :param y_score: (list or numpy.array) 正样本概率
    :param pos_label: (int) 正样本标签, 默认是None
    :return:
    """
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)

    # sensitivity = tpr
    # 1-specificity = fpr
    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)  # 计算真正率和假正率
    RightIndex = tpr + (1-fpr) - 1
    index = np.argmax(RightIndex)
    # RightIndex_val = RightIndex[index]
    # tpr_val = tpr[index]
    # fpr_val = fpr[index]
    threshold_val = threshold[index]
    return threshold_val


class GetRecallScore:
    def __init__(self, average='binary'):
        self.average = average

    def __call__(self, y_true, y_score):
        return get_recall_score(y_true, y_score)


class GetPrecisionScore:
    def __init__(self, average='binary'):
        self.average = average

    def __call__(self, y_true, y_score):
        return get_recall_score(y_true, y_score)


class GetF1Score:
    def __init__(self, average='binary'):
        self.average = average

    def __call__(self, y_true, y_score):
        return get_recall_score(y_true, y_score)
