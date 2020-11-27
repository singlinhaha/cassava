import os
import glob
from torch.utils.data import Dataset
from PIL import Image
import json
import pandas as pd
import numpy as np
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def read_label(lable_path, class_path):
    # 读取类别文件
    if class_path is not None:
        with open(class_path, "r", encoding="utf8") as fp:
            idx_to_class = json.load(fp)
            idx_to_class = dict(zip([int(i) for i in idx_to_class.keys()], idx_to_class.values()))
    else:
        idx_to_class = None

    if lable_path is not None:
        # 读取label文件
        df = pd.read_csv(lable_path)
        img_to_idx = dict(zip(list(df["image_id"]), list((df["label"]))))
    else:
        img_to_idx = None
    return img_to_idx, idx_to_class


def find_classes(dir):
    """
    获取指定目录下所有子目录的名字，并生成类别查询标签字典

    Args:
        dir (string): 文件目录
    Returns:
        classes(list): 类别名字(即子目录名字)
        class_to_idx(dict): 类别查询标签字典
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def has_file_allowed_extension(filename, extensions):
    """
    检查文件是否为允许的扩展名

    Args:
        filename (string): 文件名路径
        extensions(list or tuple): 扩展名列表

    Returns:
        bool: 如果文件名以已知的图像扩展名结尾，则为True
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions, class_to_idx=None, iter_mode="subdirectories"):
    """
    获取指定目录下所有文件名路径

    Args:
        dir (string): 文件目录
    extensions(list or tuple): 扩展名列表
    class_to_idx(dict): 类别查询标签字典，如果不为空，返回结果为(path, label),如果为空，返回结果为
                path
    iter_mode(str): 遍历模式，默认是subdirectories
            ——subdirectorie: 遍历该目录下的所有子目录，且class_to_idx的键名必须是子目录名
            ——current: 遍历该目录下的所有文件，且class_to_idx的键名必须是文件名(不带扩展名)

    Returns:
        imgaes(list): 文件路径
    """
    images = []
    dir = os.path.expanduser(dir)
    if iter_mode == "subdirectories":
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if has_file_allowed_extension(fname, extensions):
                        path = os.path.join(root, fname)
                        if class_to_idx is not None:
                            item = (path, class_to_idx[target])
                        else:
                            item = path
                        images.append(item)

    elif iter_mode == "current":
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                if has_file_allowed_extension(target, extensions):
                    if class_to_idx is not None:
                        item = (d, class_to_idx[os.path.basename(target)])
                    else:
                        item = d
                    images.append(item)

    return images


def split_dataset(root, train_percent=0.9, shuffle=True, label=None):
    # 判断label是否空，为None则使用文件夹名作为类名，否则则根据传进去label和类别
    if label is None:
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, IMG_EXTENSIONS, class_to_idx, iter_mode="subdirectories")
        samples = [(os.path.basename(path), label) for (path, label) in samples]
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
    else:
        img_to_idx, idx_to_class=read_label(label["label_path"], label["classes_path"])
        classes = list(idx_to_class.values())
        samples = [(k, v) for k, v in img_to_idx.items()]

    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                            "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
    if shuffle:
        random.shuffle(samples)

    # 获取每个类别的img索引
    class_index = dict([(k, []) for k in classes])
    for index, (path, idx) in enumerate(samples):
        class_index[idx_to_class[idx]].append(index)

    # 打乱每个类别的img索引
    for k, v in class_index.items():
        random.shuffle(v)

    # 分割数据集
    dataset = {"train": [], "val": []}
    for k, v in class_index.items():
        dataset["train"].extend([os.path.basename(samples[i][0])
                                 for i in class_index[k][:int(len(class_index[k])*train_percent)]])
        dataset["val"].extend([os.path.basename(samples[i][0])
                               for i in class_index[k][int(len(class_index[k])*train_percent):]])
    return dataset


def k_fold_split(root, n_split=3, shuffle=True, label=None):
    """
    数据集的k折划分

    Args:
        root (string): 文件目录
        n_split(int): 划分的折数
        shuffle(True): 是否在划分数据集前打乱数据集,默认是True
        label: 如果是None, 则说明使用文件夹名作为类名, 否则则是传进去label和类别

    Returns:
        k_fold_list(list): 划分好的k折数据列表
    """
    # 判断label是否空，为None则使用文件夹名作为类名，否则则根据传进去label和类别
    if label is None:
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, IMG_EXTENSIONS, class_to_idx, iter_mode="subdirectories")
        samples = [(os.path.basename(path), label) for (path, label) in samples]
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
    else:
        img_to_idx, idx_to_class=read_label(label["label_path"], label["classes_path"])
        classes = list(idx_to_class.values())
        samples = [(k, v) for k, v in img_to_idx.items()]

    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                            "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
    # 是否打乱数据集
    if shuffle:
        random.shuffle(samples)

    # 计算每个类别的数量和路径
    class_num = dict(zip(classes, [0]*len(classes)))
    class_name = dict([(k, []) for k in classes])
    for (path, idx) in samples:
        class_num[idx_to_class[idx]] += 1
        class_name[idx_to_class[idx]].append(path)

    # 数据集划分k折
    min_num = min(class_num.values())
    class_num_sclae = dict(zip(classes, [round(class_num[k]/min_num) for k in classes]))
    num_sclae = sum(list(class_num_sclae.values()))
    fold_num = len(samples) // n_split
    fold_list = []
    class_use_num = dict(zip(classes, [0] * len(classes)))
    for i in range(n_split-1):
        fold = []
        for k in classes:
            fold.extend(class_name[k][i*int(round(fold_num*(class_num_sclae[k]/num_sclae))):(i+1)*int(
                round(fold_num*(class_num_sclae[k]/num_sclae)))])
            class_use_num[k] += int(round(fold_num*(class_num_sclae[k]/num_sclae)))
        fold_list.append(fold)

    fold = []
    for k in classes:
        fold.extend(class_name[k][class_use_num[k]:])
    fold_list.append(fold)

    # k折数据组成训练集和验证集
    k_fold_list = []
    for i in range(n_split-1, -1, -1):
        val_fold = fold_list[i]
        train_fold = []
        for j in range(n_split):
            if j != i:
                train_fold.extend(fold_list[j])
        k_fold_list.append((train_fold, val_fold))

    return k_fold_list


class ImageSelectFolder(Dataset):
    """
    数据读取类
    Args:
        root (string): 文件目录
        label(None or Dict): 如果是None, 则说明使用文件夹名作为类名, 否则则是传进去label和类别
        trans(torchvision.transforms): 图像转换器
        select_condition(list): 筛选条件, 值为文件路径, 默认是None
        phase(str): 根据不同阶段返回不同的值，
    """
    def __init__(self, root, label=None, transform=None, select_condition=None,
                 data_expansion=False, phase="train"):
        self.phase = phase

        if self.phase in ["train", "val"]:
            if label is None:
                self.classes, class_to_idx = find_classes(root)
                self.samples = make_dataset(root, IMG_EXTENSIONS, class_to_idx, iter_mode="subdirectories")
                self.idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))
            else:
                img_to_idx, self.idx_to_class = read_label(label["label_path"], label["classes_path"])
                self.classes = list(self.idx_to_class.values()) if self.idx_to_class is not None else None
                self.samples = [(os.path.join(root, k), v) for k, v in img_to_idx.items()] if img_to_idx is not None else None
            if len(self.samples) == 0 or self.samples is None:
                raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                   "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        elif self.phase == "test":
            img_to_idx, self.idx_to_class = read_label(label["label_path"], label["classes_path"])
            self.classes = list(self.idx_to_class.values()) if self.idx_to_class is not None else None
            self.samples = [(i, None) for i in sorted(glob.glob(os.path.join(root, "*")))]
        self.transform = transform

        # 对数据集进行筛选
        if select_condition:
            new_sample = []
            for (path, label) in self.samples:
                if os.path.basename(path) in select_condition:
                    new_sample.append((path, label))
            self.samples = new_sample

        # 统计各个类别数量
        if self.phase != "test":
            self.class_num = self._static_class_num()

        # 是否对数据进行平衡采样
        if data_expansion:
            self._DataExpansion()

    def _static_class_num(self):
        # 统计各个类别数量
        class_num = dict(zip(self.classes, [0] * len(self.classes)))
        for (path, idx) in self.samples:
            class_num[self.idx_to_class[idx]] += 1
        return class_num

    def _DataExpansion(self):
        # 对数据进行平衡采样
        class_name = dict([(k, []) for k in self.classes])
        for (path, idx) in self.samples:
            class_name[self.idx_to_class[idx]].append((path, idx))

        sample = []
        max_num = max(self.class_num.values())
        for k, v in self.class_num.items():
            if v < max_num:
                path = class_name[k]
                times = max_num // len(path)
                remainder = max_num % len(path)
                sample.extend([path[i] for i in np.random.choice(range(len(path)), remainder)]
                              + path*times)
            else:
                sample.extend(class_name[k])
        self.samples = sample
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.phase == "train":
            return img, target
        elif self.phase == "val":
            return img, target, path
        elif self.phase == "test":
            return img, path


if __name__ == "__main__":
    k_fold_split("/media/biototem/Elements/lisen/haosen/competition/dataset/PET_data/train",
                 n_split=1,
                 shuffle=False)