import torch
import torchvision
import copy
from torch import nn

########################################################################################################################
# T2-FLAIR mismatch Image-level
class T2FLAIR_onlyImg(nn.Module):
    def __init__(self):
        super(T2FLAIR_onlyImg, self).__init__()
        self.resmodel1 = getattr(torchvision.models, "resnet18")(pretrained=True)
        # self.resmodel1 = getattr(torchvision.models, "resnet18")(pretrained=False)
        self.conv2 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)
        self.image_level = nn.Sequential(copy.deepcopy(self.resmodel1.bn1),
                                    copy.deepcopy(self.resmodel1.relu),
                                    copy.deepcopy(self.resmodel1.maxpool),
                                    copy.deepcopy(self.resmodel1.layer1),
                                    copy.deepcopy(self.resmodel1.layer2),
                                     copy.deepcopy(self.resmodel1.layer3),
                                     copy.deepcopy(self.resmodel1.layer4))

    def forward(self, t2_flair):
        out_t2_flair = self.conv2(t2_flair)
        out_t2_flair = self.image_level(out_t2_flair)

        return out_t2_flair

########################################################################################################################
# T2-FLAIR mismatch Feature-level
class Channel_Max_Pooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Channel_Max_Pooling, self).__init__()
        self.max_pooling = nn.MaxPool2d(
                    kernel_size=kernel_size,
                    stride=stride
                )
    def forward(self, x):
        x = x.transpose(1, 3)  # (batch_size, chs, h, w) -> (batch_size, w, h, chs)
        x = self.max_pooling(x)
        out = x.transpose(1, 3)
        return out


class Channel_avg_Pooling(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Channel_avg_Pooling, self).__init__()
        self.avg_pooling = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride
        )
    def forward(self, x):
        x = x.transpose(1, 3)  # (batch_size, chs, h, w) -> (batch_size, w, h, chs)
        x = self.avg_pooling(x)
        out = x.transpose(1, 3)
        return out

class T2FLAIR_onlyFea(nn.Module):
    def __init__(self):
        super(T2FLAIR_onlyFea, self).__init__()
        self.resmodel1 = getattr(torchvision.models, "resnet18")(pretrained=True)
        # self.resmodel1 = getattr(torchvision.models, "resnet18")(pretrained=False)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.relu2 = nn.ReLU(inplace=True)

        self.feature_level = nn.Sequential(copy.deepcopy(self.resmodel1.bn1),
                                    copy.deepcopy(self.resmodel1.relu),
                                    copy.deepcopy(self.resmodel1.maxpool),
                                    copy.deepcopy(self.resmodel1.layer1),
                                    copy.deepcopy(self.resmodel1.layer2),
                                     copy.deepcopy(self.resmodel1.layer3),
                                     copy.deepcopy(self.resmodel1.layer4))

        self.max_pool = Channel_Max_Pooling((1, 512), (1, 1))
        self.avg_pool = Channel_avg_Pooling((1, 512), (1, 1))
        self.sigmoid = nn.Sigmoid()

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, t2, flair):
        out_t2 = self.conv1(t2)
        out_t2 = self.feature_level(out_t2)

        out_flair = self.conv1(flair)
        out_flair = self.feature_level(out_flair)

        out_diff = out_t2 - out_flair

        check_out_diff_nan = torch.isnan(out_diff).sum()
        if check_out_diff_nan > 0:
            print("network check_out_diff_nan")

        out_max = self.max_pool(out_diff)
        out_avg = self.avg_pool(out_diff)

        out_cat = torch.cat((out_max, out_avg), dim=1)
        out_cat = self.conv2(out_cat)
        out_cat = self.relu2(out_cat)
        out_cat = self.sigmoid(out_cat)

        out_t2_aug = out_cat * out_t2
        out_flair_aug = out_cat * out_flair

        out_t2_res = out_t2 + out_t2_aug
        out_flair_res = out_flair + out_flair_aug

        out_subtraction = torch.cat((out_t2_res, out_flair_res), dim=1)

        check_out_subtraction_nan = torch.isnan(out_subtraction).sum()
        if check_out_subtraction_nan > 0:
            print("network check_out_subtraction_nan")

        return out_t2_res, out_flair_res