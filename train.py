import os
import time
import pickle
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.model import get_model
# from model.loss_function import LSR
from config import cfg
from utils.callback import ModelCheckpoint, SaveEpochMetrics
from utils.train_fit_generators import TrainFitGenerator
from utils.plot_utils import plot_training_metrics
from utils.data_reader_utils import k_fold_split, split_dataset, ImageSelectFolder
from utils.nn_metrics import GetRecallScore, GetPrecisionScore, GetF1Score
import warnings
warnings.filterwarnings("ignore")


def call_backs(file_path, history_path):
    checkpoint = ModelCheckpoint(filepath=file_path, monitor="val_acc", verbose=1,
                                 save_best_only=True, mode="max", save_weights_only=True,
                                 period=1)
    save_history = SaveEpochMetrics(filepath=history_path, period=1)
    # exp_lr_scheduler = ExpLrScheduler(init_lr=init_lr, lr_decay_epoch=10,
    #                                   weight_decay=0.8)
    return [checkpoint, save_history]


def train_transfer_learning(model, loss_function, train_dataload, val_dataload, lr=0.001,
                            epochs=5, callbacks=None, metrics=None):
    print("开始迁移学习:")
    for k,v in list(model.named_parameters())[:-2]:
        v.requires_grad = False

    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    fit_generator = TrainFitGenerator(net=model, optimizer=optimizer, loss_function=loss_function,
                                      generator=train_dataload, epochs=epochs,
                                      validation_data=val_dataload,
                                      callbacks=callbacks,
                                      metrics=metrics)
    fit_generator.run()


def trrain_fine_tuning(model, loss_function, train_dataload, val_dataload, history_save_dir,
                       lr=0.001, epochs=3, callbacks=None, metrics=None):
    print("开始微调:")
    for k, v in model.named_parameters():
        v.requires_grad = True

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    fit_generator = TrainFitGenerator(net=model, optimizer=optimizer, loss_function=loss_function,
                                      generator=train_dataload, epochs=epochs,
                                      validation_data=val_dataload,
                                      callbacks=callbacks,
                                      metrics=metrics)
    fit_generator.run()
    plot_training_metrics(fit_generator.history, history_save_dir, "loss",
                          title=f"train and validation loss", is_show=False)
    plot_training_metrics(fit_generator.history, history_save_dir, "acc",
                          title=f"train and validation accuracy", is_show=False)
    # plot_training_metrics(fit_generator.history, history_save_dir, "f1_score",
    #                       title=f"train and validation f1", is_show=False)


def main(cfg, step):
    model_save_dir = os.path.join(cfg["output_dir"], "weights"+step)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    history_save_dir = os.path.join(cfg["output_dir"], "visual"+step)
    if not os.path.exists(history_save_dir):
        os.makedirs(history_save_dir)

    # time_mark = '2020_03_06_'
    # 以当前时间作为保存的文件名标识
    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))
    file_path = os.path.join(model_save_dir, time_mark + "epoch_{epoch}-model_weights.pth")
    history_path = os.path.join(history_save_dir, time_mark + "result.csv")
    callbacks_s = call_backs(file_path, history_path)

    train_dataset = ImageSelectFolder(root=cfg["train_dataset"],
                                      label=cfg["label"],
                                      select_condition=cfg["train_select"],
                                      data_expansion=True,
                                      transform=transforms.Compose([
                                          transforms.RandomApply([transforms.RandomCrop(size=(448, 448)),
                                                                  # transforms.RandomResizedCrop(size=cfg["img_width"]),
                                                                  ],
                                                                 p=0.3),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(360),
                                          transforms.Resize((cfg["img_width"], cfg["img_hight"])),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                          ]))
    val_dataset = ImageSelectFolder(root=cfg["val_dataset"],
                                    label=cfg["label"],
                                    select_condition=cfg["val_select"],
                                    transform=transforms.Compose([
                                        transforms.Resize((cfg["img_width"], cfg["img_hight"])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                        ]))

    train_dataload = DataLoader(dataset=train_dataset, batch_size=cfg["batch_size"],
                                shuffle=True)
    val_dataload = DataLoader(dataset=val_dataset, batch_size=cfg["batch_size"],
                              shuffle=False)

    model = get_model(model_weight_path=cfg["model_weight_path"],
                      model_name=cfg["model_name"],
                      out_features=cfg["num_classes"],
                      img_width=cfg["img_width"],
                      img_hight=cfg["img_hight"],
                      verbose=True)
    model.cuda()
    loss_function = nn.CrossEntropyLoss().cuda()

    # 定义额外的评价指标
    recall = GetRecallScore(average="micro")
    precision = GetPrecisionScore(average="micro")
    f1 = GetF1Score(average="micro")
    metrics = {"recall": recall, "precision": precision, "f1 score": f1}

    train_transfer_learning(model, loss_function, train_dataload, val_dataload,
                            cfg["tl_lr"], epochs=3, metrics=metrics)
    trrain_fine_tuning(model, loss_function, train_dataload, val_dataload, history_save_dir,
                       lr=cfg["ft_lr"], epochs=cfg["nepochs"],
                       callbacks=callbacks_s, metrics=metrics)
    del model


def train(cfg, step):
    # 设置保存目录
    model_save_dir = os.path.join(cfg["output_dir"], "weights" + step)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    history_save_dir = os.path.join(cfg["output_dir"], "visual" + step)
    if not os.path.exists(history_save_dir):
        os.makedirs(history_save_dir)

    # time_mark = '2020_03_06_'
    # 以当前时间作为保存的文件名标识
    time_mark = time.strftime('%Y_%m_%d_', time.localtime(time.time()))
    file_path = os.path.join(model_save_dir, time_mark + "epoch_{epoch}-model_weights.pth")
    history_path = os.path.join(history_save_dir, time_mark + "result.csv")
    callbacks_s = call_backs(file_path, history_path)

    # 加载数据集
    train_dataset = ImageSelectFolder(root=cfg["train_dataset"],
                                      label=cfg["label"],
                                      select_condition=cfg["train_select"],
                                      data_expansion=True,
                                      transform=transforms.Compose([
                                          transforms.RandomApply([transforms.RandomCrop(size=(448, 448)),
                                                                  # transforms.RandomResizedCrop(size=cfg["img_width"]),
                                                                  ],
                                                                 p=0.3),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.RandomRotation(360),
                                          transforms.Resize((cfg["img_width"], cfg["img_hight"])),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                          ]))
    val_dataset = ImageSelectFolder(root=cfg["val_dataset"],
                                    label=cfg["label"],
                                    select_condition=cfg["val_select"],
                                    transform=transforms.Compose([
                                        transforms.Resize((cfg["img_hight"], cfg["img_width"])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]))

    train_dataload = DataLoader(dataset=train_dataset, batch_size=cfg["batch_size"],
                                shuffle=True)
    val_dataload = DataLoader(dataset=val_dataset, batch_size=cfg["batch_size"],
                              shuffle=False)

    model = get_model(model_weight_path=cfg["model_weight_path"],
                      model_name=cfg["model_name"],
                      out_features=cfg["num_classes"],
                      img_width=cfg["img_width"],
                      img_hight=cfg["img_hight"],
                      verbose=False)
    model.cuda()
    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    fit_generator = TrainFitGenerator(net=model, optimizer=optimizer, loss_function=loss_function,
                                      generator=train_dataload, epochs=cfg["nepochs"],
                                      validation_data=val_dataload,
                                      callbacks=callbacks_s)
    fit_generator.run()
    plot_training_metrics(fit_generator.history, history_save_dir, "loss",
                          title=f"train and validation loss", is_show=False)
    plot_training_metrics(fit_generator.history, history_save_dir, "acc",
                          title=f"train and validation accuracy", is_show=False)
    # plot_training_metrics(fit_generator.history, history_save_dir, "f1_score",
    #                       title=f"train and validation f1", is_show=False)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not os.path.exists(cfg["output_dir"]):
        os.makedirs(cfg["output_dir"])
    if cfg["fold"] == 1 or cfg["fold"] is None:
        if 0 < cfg["train_percent"] < 1:
            cfg["train_dataset"] = cfg["dataset"]
            cfg["val_dataset"] = cfg["dataset"]
            dataset = split_dataset(root=cfg["dataset"], train_percent=cfg["train_percent"],
                                    shuffle=True, label=cfg["label"])
            cfg["train_select"] = dataset["train"]
            cfg["val_select"] = dataset["val"]
            with open(os.path.join(cfg["output_dir"],
                                   "dataset_{}-split_train_val.pkl".format(cfg["train_percent"])),
                      "wb") as fp:
                pickle.dump([dataset["train"], dataset["val"]], fp)

        step = ""
        main(cfg, step)

    elif cfg["fold"] > 1:
        k_fold_list = k_fold_split(root=cfg["dataset"], n_split=cfg["fold"],
                                   shuffle=True, label=cfg["label"])
        cfg["train_dataset"] = cfg["dataset"]
        cfg["val_dataset"] = cfg["dataset"]
        with open(os.path.join(cfg["output_dir"],
                               "dataset_{}-fold_train_val.pkl".format(cfg["fold"])),
                  "wb") as fp:
            pickle.dump(k_fold_list, fp)
        for i, (train_select,val_select) in enumerate(k_fold_list):
            cfg["train_select"] = train_select
            cfg["val_select"] = val_select
            step = "_{i}".format(i=i)
            main(cfg, step)