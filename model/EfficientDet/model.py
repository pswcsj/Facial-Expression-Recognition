import torch
from torch import nn
from torch.nn import functional as F
from model.EfficientNet import EfficientNet


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        model = EfficientNet.from_pretrained('EfficientNet-b7')
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.

        # 결국 feature_maps에는 resolutin reduction이 진행되기 전의 feature 5개, 맨 마지막 블록장의 output 1개 저장
        # 처음 것 제외하고 5개만
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:] # p2, p3, p4, p5 반환

class BiFPN(nn.Module):
    def __init__(self):
        super(BiFPN, self).__init__()
