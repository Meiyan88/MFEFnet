import torch.nn.functional as F
import torch
from torch import nn, einsum
# from torchviz import make_dot
import os
# import graphviz
from torch.autograd import Function
from Network.SFE_module import Unet_based_model
from Network.DFF_module import MultiHeadAttention, MIL_Attention
from Network.AMF_module import T2FLAIR_onlyFea,T2FLAIR_onlyImg

class MFEFnet(nn.Module):
    def __init__(self):
        super(MFEFnet, self).__init__()
        self.SFE_module = Unet_based_model(
                                encoder_name='se_resnet50',
                                classes=2,
                                in_channels=4,
                                encoder_weights="imagenet")
        self.TF_Img = T2FLAIR_onlyImg()
        self.TF_Fea = T2FLAIR_onlyFea()
        self.Intra_slice = MultiHeadAttention(n_head=4, d_model=512,
                                                       d_k=512 // 4,
                                                       d_v=512 // 4,
                                                       use_residual=False)
        self.Inter_slice = MIL_Attention()

        self.wh1 = nn.Linear(2048, 512)
        self.leaky_relu1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(512)

        self.wh2 = nn.Linear(2048, 1024)
        self.leaky_relu2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(1024)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.header_class = nn.Sequential(
        #     nn.Linear(2048, 2048, bias=False),
        #     nn.Dropout(p=0.4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(2048, 1, bias=True),
        # )

    def forward(self, image, image_t2flair, image_t2, image_flair, batch_index):

        # extract tumor_related features
        masks_tumor, features_tumor_list = self.SFE_module(image)

        # extract image-level features
        features_img = self.TF_Img(image_t2flair)

        # extract feature-level features
        features_T2, features_FLAIR = self.TF_Fea(image_t2, image_flair)

        # transform tumor related features
        features_tumor = self.avgpool(features_tumor_list[-1]).view(features_tumor_list[-1].size(0), -1)
        features_tumor = self.wh1(features_tumor)
        features_tumor = self.bn1(features_tumor)
        features_tumor = self.leaky_relu1(features_tumor)

        # transform T2 FLAIR mismatch related features
        features_img = self.avgpool(features_img).view(features_img.size(0), -1)
        features_T2 = self.avgpool(features_T2).view(features_T2.size(0), -1)
        features_FLAIR = self.avgpool(features_FLAIR).view(features_FLAIR.size(0), -1)

        # feature fusion
        feature_concat = torch.cat([features_tumor.unsqueeze(1), features_img.unsqueeze(1), features_T2.unsqueeze(1),
                                    features_FLAIR.unsqueeze(1)], dim=1)

        feature_fin, _ = self.Intra_slice(feature_concat, feature_concat, feature_concat)
        feature_fin = feature_fin.view(feature_fin.size(0), -1)

        # out_c = feature_fin.view(feature_fin.size(0), -1)
        # result = self.header_class(feature_fin.view(feature_fin.size(0), -1))

        # return out_c, masks_tumor

        feature_fin = self.wh2(feature_fin)
        feature_fin = self.bn2(feature_fin)
        out_c = self.leaky_relu2(feature_fin)

        if len(batch_index) != 0:
            out_c_list = []
            for i, j in enumerate(batch_index):
                if i < (len(batch_index) - 2) or i == (len(batch_index) - 2):
                    out_c_sub = out_c[j:batch_index[i + 1], :]
                    # out_c_sub = self.layer_norm(out_c_sub)
                    out_c_sub = self.Inter_slice(out_c_sub)
                    out_c_list.append(out_c_sub)
            # out_c, out_mask = self.feature_extractor(x1, x2)
            out_c = torch.cat(out_c_list, dim=0)
        else:
            # out_c = self.layer_norm(out_c)
            # out_c, A = self.attention(out_c)
            out_c = self.Inter_slice(out_c)

        return out_c, masks_tumor

if __name__ == '__main__':
    T2 = torch.ones((2, 1, 224, 224))
    FLAIR = torch.ones((2, 1, 224, 224))
    Image = torch.ones((2, 4, 224, 224))
    # batch_index indicates the
    batch_index = []

    model = MFEFnet()
    output = model(image=Image, image_t2flair=T2-FLAIR, image_t2=T2, image_flair=FLAIR, batch_index=batch_index)
    print()
