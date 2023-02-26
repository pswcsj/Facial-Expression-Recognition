import torch
import torch.nn as nn
from model.EfficientNet import MBConvBlock


def load_pretrained(model, weights_path):
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)

    #    Conv 아니면 Bn으로 이루어져 있다. 그리고 model.modules()의 개수가 state_dict의 개수보다 더 적다.
    # module을 순회하면서 만약, module이 Conv2d라면, dict의 key에 conv라는 문자열이 포함되지 않을 때까지 dict를 계속 앞에서부터 제거하며
    # split('.')[-1]로 가장 마지막 것(bias, weight) 등을 찾은 다음 module[att] = weight를 주어 파라미터를 업데이트 시킨다.
    # 가장 마지막에 state_dict가 빈 배열이 되었는지를 확인하여 마무리.
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            key, value = state_dict.popitem(last=False)
            point = key.split('.')[:-1].join('.')
            while point in key:
                if 'weight' in key or 'bias' in key:
                    setattr(module, key.split('.')[-1], nn.Parameter(value.float()))
                else:
                    setattr(module, key.split('.')[-1], value.float())
                if state_dict:
                    key, value = state_dict.popitem(last=False)
                else:
                    break

    assert not state_dict
    print('loaded')