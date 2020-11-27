import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, densenet121, densenet161
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
from collections import OrderedDict


def get_model(model_weight_path=None, model_name="resnet18", out_features=2, img_width=224, img_hight=224,
              verbose=True):
    if model_name == "resnet18":
        model = resnet18(pretrained=False)
        if model_weight_path is not None:
            model.load_state_dict(torch.load(model_weight_path))
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in, out_features)
    elif model_name == "resnet50":
        model = resnet50(pretrained=False)
        if model_weight_path is not None:
            model.load_state_dict(torch.load(model_weight_path))
        num_fc_in = model.fc.in_features
        # model.add_module("dropout", nn.Dropout(p=0.5))
        model.fc = nn.Linear(num_fc_in, out_features)

    elif model_name == "densenet121":
        model = densenet121(pretrained=False)
        if model_weight_path is not None:
            # model.load_state_dict(torch.load(model_weight_path))

            state_dict =torch.load(model_weight_path)
            # 初始化一个空 dict
            new_state_dict = OrderedDict()
            # 修改 key
            for k, v in state_dict.items():
                if 'denseblock' in k:
                    param = k.split(".")
                    k = ".".join(param[:-3] + [param[-3]+param[-2]] + [param[-1]])
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        num_classifier_in = model.classifier.in_features
        model.classifier = nn.Linear(num_classifier_in, out_features)

    elif model_name == "densenet161":
        model = densenet161(pretrained=False)
        if model_weight_path is not None:
            state_dict = torch.load(model_weight_path)
            # 初始化一个空 dict
            new_state_dict = OrderedDict()
            # 修改 key
            for k, v in state_dict.items():
                if 'denseblock' in k:
                    param = k.split(".")
                    k = ".".join(param[:-3] + [param[-3] + param[-2]] + [param[-1]])
                new_state_dict[k] = v
            model.load_state_dict(new_state_dict)

        num_classifier_in = model.classifier.in_features
        model.classifier = nn.Linear(num_classifier_in, out_features)

    elif model_name in ["efficientnet-b0", "efficientnet-b2"]:
        model = EfficientNet.from_name(model_name)
        if model_weight_path is not None:
            model.load_state_dict(torch.load(model_weight_path))
        num_fc_in = model._fc.in_features
        model._fc = nn.Linear(num_fc_in, out_features)

    if verbose is True:
        model.cpu().eval()
        summary(model, input_size=(3, img_hight, img_width), device="cpu")
    return model