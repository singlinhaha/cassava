import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import transforms
from utils.data_reader_utils import has_file_allowed_extension, make_dataset, IMG_EXTENSIONS


class GradCAM(object):
    def __init__(self, model, feature_layer, img_width, img_hight, idx_to_class,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), use_gpu=True):
        self.model = model
        self.model.eval()
        self.feature_layer = feature_layer
        self.img_width = img_width
        self.img_hight = img_hight
        self.idx_to_class = idx_to_class
        self.fmap_block = []
        self.grad_block = []
        self.trans = transforms.Compose([transforms.Resize((self.img_width, self.img_hight)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
        self.feature_layer.register_forward_hook(self._forward_hook)
        self.feature_layer.register_backward_hook(self._backward_hook)
        if use_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)

    def _open_img(self,img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.trans(img)
        img = img.unsqueeze(0)
        return img

    # 注册hook
    def _forward_hook(self, module, input, output):
        self.fmap_block.append(output)

    # 注册hook
    def _backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def computerClassVec(self, output, index=None):
        """
        计算类向量
        :param ouput_vec: tensor
        :param index: int，指定类别
        :return: tensor
        """
        if index is None:
            index = np.argmax(output.cpu().detach().numpy())
        else:
            index = np.array(index)

        index = index[np.newaxis, np.newaxis]
        index = torch.from_numpy(index)
        one_hot = torch.zeros(output.size()).scatter_(1, index, 1)
        one_hot = one_hot.to(self.device)
        one_hot.requires_grad = True
        class_vec = torch.sum(one_hot * output)
        return class_vec

    def gen_cam(self, feature_map, grads):
        """
        依据梯度和特征图，生成cam
        :param feature_map: np.array， in [C, H, W]
        :param grads: np.array， in [C, H, W]
        :return: np.array, [H, W]
        """
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
        weights = np.mean(grads, axis=(1, 2))  # weights shape(C,)

        for i, w in enumerate(weights):
            cam += feature_map[i,:,:] * w

        cam = np.maximum(cam, 0)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

    def getCAM(self, img_path):
        img = self._open_img(img_path)
        img = img.to(self.device)

        # forward
        output = self.model(img)
        idx = np.argmax(output.cpu().detach().numpy())

        # backward
        self.model.zero_grad()
        class_loss = self.computerClassVec(output)
        class_loss.backward()

        # 生成cam
        grad_value = self.grad_block[0].cpu().detach().numpy().squeeze()
        fmap = self.fmap_block[0].cpu().detach().numpy().squeeze()
        cam = self.gen_cam(fmap, grad_value)
        self.grad_block, self.fmap_block = [], []
        return cam, idx

    def createCAM(self, root, img_to_idx, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(root)

        frame_path = []
        if os.path.isfile(root):
            frame = os.path.basename(root)
            if has_file_allowed_extension(frame, IMG_EXTENSIONS):
                frame_path.append(root)
        elif os.path.isdir(root):
            frame_path = make_dataset(dir=root, extensions=IMG_EXTENSIONS,
                                      iter_mode="current")

        for path in frame_path:
            name = os.path.basename(path).split(".")[0]
            label = img_to_idx[name]
            cam, idx = self.getCAM(path)

            img = cv2.imread(path)
            img = cv2.resize(img, dsize=(self.img_width, self.img_hight))
            heatmap = cv2.resize(cam, dsize=(self.img_width, self.img_hight))
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, colormap=cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(img, 0.8, heatmap, 0.2, 0)

            if idx == label:
                cv2.imwrite(os.path.join(save_dir, "{}_{}_to_{}_heatmap.jpg".format(name, label, idx)),
                            heatmap)
                cv2.imwrite(os.path.join(save_dir, "{}_{}_to_{}_superimposed.jpg".format(name, label, idx)),
                            superimposed_img)


if __name__ == "__main__":
    from model.model import get_model
    from config import cfg

    root = "/media/biototem/Elements/lisen/haosen/competition/dataset/PET_data/val/AD"
    img_to_idx = dict(zip([i.split(".")[0] for i in
                           os.listdir(root)], [0] * len(os.listdir(root))))
    save_dir = "/media/biototem/Elements/lisen/haosen/competition/PET/output/densenet121_fold=3_epoch=80/AD_true"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model = get_model(model_weight_path=None,
                      model_name=cfg["model_name"],
                      out_features=cfg["num_classes"],
                      img_width=cfg["img_width"],
                      img_hight=cfg["img_hight"],
                      verbose=True)
    model.load_state_dict(torch.load(cfg["model_weights"][0]))
    cam = GradCAM(model=model, feature_layer=model.features, img_width=cfg["img_width"], img_hight=cfg["img_hight"],
                  idx_to_class=cfg["idx_to_class"])
    cam.createCAM(root=root,
                  img_to_idx=img_to_idx,
                  save_dir=save_dir)