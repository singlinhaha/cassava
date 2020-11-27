import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import csv
import pickle
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils.data_reader_utils import ImageSelectFolder
from model.model import get_model
from utils.nn_metrics import get_auc_score, get_recall_score, get_precision_score, get_f1_score
from utils.nn_metrics import get_accuracy_score, get_average_precision_score, get_roc_best_threshold
from utils.plot_utils import plot_confusion_matrix
from config import cfg


class Prediction(object):
    def __init__(self, cfg):
        self.dataset_root = cfg["test_dataset"]
        self.batch_size = cfg["batch_size"]
        self.img_width = cfg["img_width"]
        self.img_hight = cfg["img_hight"]
        self.model_weights = cfg["model_weights"]
        self.phase = cfg["phase"]
        self.save_dir = os.path.join(cfg["save_dir"], "test")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.model = get_model(model_weight_path=None,
                               model_name=cfg["model_name"],
                               out_features=cfg["num_classes"],
                               img_width=cfg["img_width"],
                               img_hight=cfg["img_hight"],
                               verbose=True)

        self.judge_model_weight_path()

    def judge_model_weight_path(self):
        if isinstance(self.model_weights, list) and len(self.model_weights) > 0:
            self.model_weights_path = self.model_weights
        elif isinstance(self.model_weights, str) and os.path.isfile(self.model_weights):
            self.model_weights_path = [self.model_weights]
        elif isinstance(self.model_weights, str) and os.path.isdir(self.model_weights):
            self.model_weights_path = glob.glob(os.path.join(self.model_weights, "*.pth"))
        else:
            raise ValueError("model_weights set error!")

    def load_model_weight(self, model_weight):
        self.model.load_state_dict(torch.load(model_weight)["state_dict"])
        self.model.cuda().eval()

    def get_statistics_title(self, k_fold=False):
        statistics_title = []
        statistics_title.append("slide id")
        statistics_title.append("positive probability")
        statistics_title.append("predicted value")
        statistics_title.append("true value")

        if k_fold:
            statistics_title.append("")
            statistics_title.append("frist positive probability")
            statistics_title.append("second positive probability")
            statistics_title.append("third positive probability")
        return statistics_title

    def val_predict(self):
        if cfg["select_file"] is None:
            select_condition = None
        else:
            with open(cfg["select_file"], "rb") as fp:
                select_condition = pickle.load(fp)
            if isinstance(select_condition[0][0], str):
                select_condition = select_condition[1]
            elif isinstance(select_condition[0][0], list):
                select_condition = [i[1] for i in select_condition]
        dataset = ImageSelectFolder(root=cfg["test_dataset"],
                                    label=cfg["test_label"],
                                    select_condition=select_condition,
                                    transform=transforms.Compose([
                                        transforms.Resize((cfg["img_width"], cfg["img_hight"])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]),
                                    phase="val")

        dataload = DataLoader(dataset=dataset, batch_size=cfg["batch_size"],
                              shuffle=False)

        k_outcome_list = []
        k_outcome_prob_list = []
        k_positive_prob_list = []
        for model_weights in self.model_weights_path:
            self.load_model_weight(model_weights)
            self.model.eval()
            # 预测
            name_list = []
            outcome_prob_list = []
            outcome_list = []
            true_list = []
            positive_prob_list = []
            with torch.no_grad():
                with tqdm(dataload, desc='inferencing',
                          file=sys.stdout, disable=False) as iterator:
                    for count, (x, y, name) in enumerate(iterator):
                        x = x.cuda()
                        output_score = self.model(x)
                        y_pred_prob = F.softmax(output_score.detach().cpu(), dim=-1)
                        y_prob, y_pred = torch.max(y_pred_prob, dim=-1)
                        name_list.extend([os.path.basename(i) for i in name])
                        positive_prob_list.append(y_prob)
                        outcome_prob_list.append(y_pred_prob)
                        outcome_list.append(y_pred)
                        true_list.extend(y.cpu().numpy().tolist())

            k_positive_prob_list.append(torch.cat(positive_prob_list, dim=0).numpy())
            k_outcome_prob_list.append(torch.cat(outcome_prob_list, dim=0).numpy())
            k_outcome_list.append(torch.cat(outcome_list, dim=0).numpy())

        if len(k_positive_prob_list) == 1:
            finally_outcome = k_outcome_list[0]
            finally_positive_prob = k_positive_prob_list[0]
        else:
            finally_positive_prob = np.mean(np.concatenate([np.expand_dims(i, axis=1)
                                                            for i in k_positive_prob_list], axis=1), axis=-1)

        # 保存统计指标
        acc = get_accuracy_score(y_true=true_list,
                                 y_pred=finally_outcome)
        precision = get_precision_score(y_true=true_list,
                                        y_pred=finally_outcome,
                                        average="micro")
        recall = get_recall_score(y_true=true_list,
                                  y_pred=finally_outcome,
                                  average="micro")
        f1_score = get_f1_score(y_true=true_list,
                                y_pred=finally_outcome,
                                average="micro")
        # auc = get_auc_score(y_true=true_list,
        #                     y_score=finally_positive_prob)
        # average_precision = get_average_precision_score(y_true=true_list,
        #                                                 y_score=finally_positive_prob)
        # best_roc_auc_threshold = get_roc_best_threshold(y_true=true_list,
        #                                                 y_score=finally_positive_prob)

        plot_confusion_matrix(y_true=true_list,
                              y_pred=finally_outcome,
                              title="confusion matrix",
                              save_dir=self.save_dir,
                              is_show=False)

        fp_0 = open(os.path.join(self.save_dir, 'statistics.csv'), mode="w",
                    encoding='utf-8-sig', newline='')
        writer_0 = csv.writer(fp_0)
        # writer_0.writerow(["acc", "precision", "recall", "f1_score", "roc_auc", "average_precision", "best_roc_threshold"])
        # writer_0.writerow([round(acc, 4), round(precision, 4), round(recall, 4),
        #                    round(f1_score, 4), round(auc, 4), round(average_precision, 4),
        #                    best_roc_auc_threshold])
        writer_0.writerow(["acc", "precision", "recall", "f1_score"])
        writer_0.writerow([round(acc, 4), round(precision, 4), round(recall, 4),
                           round(f1_score, 4)])
        writer_0.writerow("\n")
        if len(self.model_weights_path) > 1:
            # writer_0.writerow(["acc_flod", "precision_flod", "recall_flod", "f1_score_flod",
            #                    "roc_auc_flod", "average_precision_flod"])
            writer_0.writerow(["acc_flod", "precision_flod", "recall_flod", "f1_score_flod"])
            for i in range(len(k_outcome_list)):
                acc_flod = get_accuracy_score(y_true=true_list,
                                              y_pred=k_outcome_list[i])
                precision_flod = get_precision_score(y_true=true_list,
                                                     y_pred=k_outcome_list[i],
                                                     average="micro")
                recall_flod = get_recall_score(y_true=true_list,
                                               y_pred=k_outcome_list[i],
                                               average="micro")
                f1_score_flod = get_f1_score(y_true=true_list,
                                             y_pred=k_outcome_list[i],
                                             average="micro")
                # auc_flod = get_auc_score(y_true=true_list,
                #                          y_score=k_positive_prob_list[i])
                # average_precision_flod = get_average_precision_score(y_true=true_list,
                #                                                      y_score=k_positive_prob_list[i])

                # writer_0.writerow([round(acc_flod, 4), round(precision_flod, 4), round(recall_flod, 4),
                #                    round(f1_score_flod, 4), round(auc_flod, 4), round(average_precision_flod, 4)])
                writer_0.writerow([round(acc_flod, 4), round(precision_flod, 4), round(recall_flod, 4),
                                   round(f1_score_flod, 4)])
            writer_0.writerow("\n")

            csv_title = self.get_statistics_title(True)
            writer_0.writerow(csv_title)
            for i in range(len(name_list)):
                writer_0.writerow([name_list[i], round(1-finally_positive_prob[i], 4),
                                   round(finally_positive_prob[i], 4), finally_outcome[i], true_list[i],
                                   "", round(k_positive_prob_list[0][i], 4), round(k_positive_prob_list[1][i], 4),
                                   round(k_positive_prob_list[2][i], 4)])
        else:
            csv_title = self.get_statistics_title(False)
            writer_0.writerow(csv_title)
            for i in range(len(name_list)):
                writer_0.writerow([name_list[i],
                                   round(finally_positive_prob[i], 4), finally_outcome[i], true_list[i]])
        fp_0.close()

    def test_predict(self):
        dataset = ImageSelectFolder(root=cfg["test_dataset"],
                                    label=cfg["test_label"],
                                    transform=transforms.Compose([
                                        transforms.Resize((cfg["img_width"], cfg["img_hight"])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                                    ]),
                                    phase="test")
        dataload = DataLoader(dataset=dataset, batch_size=cfg["batch_size"],
                              shuffle=False)

        k_positive_prob_list = []
        k_outcome_list = []
        for model_weights in self.model_weights_path:
            self.load_model_weight(model_weights)
            self.model.eval()
            # 预测
            name_list = []
            positive_prob_list = []
            outcome_list = []
            with torch.no_grad():
                with tqdm(dataload, desc='inferencing',
                          file=sys.stdout, disable=False) as iterator:
                    for count, (x, name) in enumerate(iterator):
                        x = x.cuda()
                        output_score = self.model(x)
                        y_pred_prob = F.softmax(output_score.detach().cpu(), dim=-1)
                        _, y_pred = torch.max(y_pred_prob, dim=-1)
                        outcome_list.append(y_pred)
                        name_list.extend([os.path.basename(i) for i in name])
                        positive_prob_list.append(y_pred_prob[:, -1])

            k_outcome_list.append(torch.cat(outcome_list, dim=0).numpy())
            k_positive_prob_list.append(torch.cat(positive_prob_list, dim=0).numpy())

        if len(k_positive_prob_list) == 1:
            finally_outcome_list = (k_outcome_list[0])
        else:
            finally_positive_prob_list = np.mean(torch.cat([i.unsqueeze_(dim=1)
                                                            for i in k_positive_prob_list], dim=1).numpy(), axis=-1)

        outcome = [dataset.idx_to_class[i] for i in finally_outcome_list]
        self.saveSubmission(name_list, outcome)

    def saveSubmission(self, name, outcome):
        submission={
            'uuid': name, 'label': outcome
        }
        df = pd.DataFrame(submission)
        df.to_csv(os.path.join(self.save_dir, "submission.csv"), index=False)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    prediction = Prediction(cfg)
    if cfg["phase"] == "val":
        prediction.val_predict()
    elif cfg["phase"] == "test":
        prediction.test_predict()