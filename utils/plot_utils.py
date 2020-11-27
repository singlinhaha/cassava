"""
该脚本实现各种画图功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
from utils.nn_metrics import data_transform


def plot_training_metrics(history, save_dir=None, plot_type="loss", title=None, is_show=False):
    """
    绘制指定指标的训练和验证曲线图, 可以选择"loss", "acc", "roc_auc", "average_precision", "precision", "recall", "f1_score"
    :param history: (dict) 指标保存字典
    :param save_dir: (str) 保存目录
    :param plot_type: (str) 想要绘制的指标
    :param title: (str) 标题, 默认是None
    :param is_show: (bool) 是否展示, 默认是Fasle
    :return:
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if plot_type not in ["loss", "acc", "roc_auc", "average_precision", "precision",
                         "recall", "f1_score"]:
        raise ValueError("plot_type error! this should be in [loss, acc, roc_auc, average_precision, precision, recall, f1_score]")
    train_plot_value = history[plot_type]
    val_plot_value = history["val_"+plot_type]
    epochs = range(1,len(train_plot_value)+1)

    plt.figure()
    plt.plot(epochs, train_plot_value, "r-", label="train")
    plt.plot(epochs, val_plot_value, "b--", label="val")
    plt.legend()
    if title is None:
        title = f"train and validation {plot_type}"
    plt.title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    if is_show:
        plt.show()


def plot_training_data(data, save_dir=None, title=None, is_show=False):
    """
    绘制数据曲线图
    :param data: (list) 指标数据
    :param save_dir: (str) 保存目录, 默认是None
    :param title: (str) 标题, 默认是None
    :param is_show: (bool) 是否展示, 默认是Fasle
    :return:
    """
    epochs = range(1, len(data) + 1)
    plt.figure()
    plt.plot(epochs, data, "b-", label="val")
    plt.legend()

    if save_dir is not None:
        if title is not None:
            plt.title(title)
            plt.savefig(os.path.join(save_dir, "{}.png".format(title)))
        else:
            plt.savefig(os.path.join(save_dir, "data.png"))
    if is_show:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, title=None, save_dir=None, is_percentage=False, is_show=False):
    """
    绘制混淆矩阵
    :param y_true: (list or numpy.array) 标签
    :param y_pred: (list or numpy.array) 预测值
    :param labels: (list) 标签列表, 默认是二分类, 即(0, 1)
    :param title: (str) 标题, 默认是None
    :param save_dir: (str) 保存目录, 默认是None
    :param is_percentage: (bool) 是否以百分比的形式, 默认是Fasle
    :param is_show: (bool) 是否展示, 默认是Fasle
    :return:
    """
    y_true = data_transform(y_true)
    y_pred = data_transform(y_pred)
    if title is None:
        title = "confusion matrix"

    matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    if is_percentage:
        matrix = (matrix.T/np.sum(matrix, axis=1)).T
        fmt = ".4g"
    else:
        fmt = ".20g"
    plt.figure()
    sns.set()
    f, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax, fmt=fmt)
    ax.set_title(title)
    ax.set_xlabel("predict")
    ax.set_ylabel("true")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, f"{title}.png"))
    if is_show:
        plt.show()
    plt.close()
    sns.reset_defaults()


def plot_roc_auc(y_true, y_score, pos_label=None, title=None, save_dir=None, is_show=False):
    """
    绘制混淆矩阵
    :param y_true: (list or numpy.array) 标签
    :param y_score: (list or numpy.array) 正样本概率
    :param pos_label: (list) 正样本标签, 默认是None
    :param title: (str) 标题, 默认是None
    :param save_dir: (str) 保存目录, 默认是None
    :param is_show: (bool) 是否展示, 默认是Fasle
    :return:
    """
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    # Compute ROC curve and ROC area for each class

    fpr, tpr, threshold = roc_curve(y_true, y_score, pos_label=pos_label)  # 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  # 计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)  # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title is None:
        plt.title('Receiver operating characteristic')
    else:
        plt.title(title)
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "Receiver operating characteristic{}.png".format(title)))
    if is_show:
        plt.show()
    plt.close()


def plot_precision_recall(y_true, y_score, pos_label=None, save_dir=None, is_show=False):
    y_true = data_transform(y_true)
    y_score = data_transform(y_score)
    # Compute PR curve and PR area for each class
    precision, recall, threshold = precision_recall_curve(y_true, y_score, pos_label=pos_label)  # 计算精确率和召回率
    pr_score = average_precision_score(y_true=y_true, y_score=y_score)  # 计算预测值的平均准确率,该分数对应于presicion-recall曲线下的面积

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision,color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % pr_score)  # 召回率为横坐标，精确率为纵坐标做曲线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    plt.legend(loc="lower right")

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "Precision-Recall.png"))
    if is_show:
        plt.show()
    plt.close()

