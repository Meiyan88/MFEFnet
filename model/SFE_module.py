import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch
from torch import nn, einsum
# from torchviz import make_dot
import os
# import graphviz

# SFE module is based on the segmentation_models_pytorch
class Unet_based_model(smp.Unet):
    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks, features

if __name__ == '__main__':

    network_tumor = Unet_based_model(
        encoder_name='se_resnet50',
        classes=2,
        in_channels=4,
        encoder_weights="imagenet")

    input = torch.ones((1, 4, 224, 224))
    out1, out2 = network_tumor(input)
    print()
