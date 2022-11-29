"""This code was written by Junshen Xu and is part of the fetal-IQA 
repository at https://github.com/daviddmc/fetal-IQA
"""
from torch import nn
import torchvision


"""pretrained models"""


class DualFC(nn.Module):
    def __init__(self, num_ftrs, num_classes, p=0, is_dual=True):
        super().__init__()
        self.is_dual = is_dual
        # self.dropout = nn.Dropout(p=p) if p else None
        self.fc1 = nn.Linear(num_ftrs, num_classes)
        # self.fc2 = nn.Linear(num_ftrs, 1) if is_dual else None

    def forward(self, x):
        # return self.fc1(x)
        x1 = self.fc1(
            x
        )  # self.fc1(self.dropout(x)) if self.dropout is not None else self.fc1(x)
        if self.is_dual:
            # x2 = self.fc2(x) #self.fc2(self.dropout(x)) if self.dropout is not None else self.fc2(x)
            return x1, x
        else:
            return x1


class PretrainedModel(nn.Module):
    def __init__(self, model_name, num_classes, p=0):
        super().__init__()
        self.model = getattr(torchvision.models, model_name)(pretrained=True)
        if model_name.startswith("vgg"):
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = DualFC(num_ftrs, num_classes, p)
        elif model_name.startswith("res"):
            num_ftrs = self.model.fc.in_features
            self.model.fc = DualFC(num_ftrs, num_classes, p)
            # self.model.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = self.model(x)
        return x


def resnet34(pretrained=True, **kwargs):
    return PretrainedModel("resnet34", num_classes=kwargs["num_classes"])
