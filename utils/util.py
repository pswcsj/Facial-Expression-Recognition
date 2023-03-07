import torch
import torch.nn as nn
from model.EfficientNet import EfficientNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#현재 모델의 ENet과 다른 모델로 만든 pt 파일을 현재 모델의 ENet에 적용하고 가중치를 저장하는 함수
# 변환된 가중치 파일은 ENetB2_VggFace2_modified.pt으로 저장됨
def load_pretrained(model, weights_path):
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    #    Conv 아니면 Bn으로 이루어져 있다. 그리고 model.modules()의 개수가 state_dict의 개수보다 더 적다.
    # module을 순회하면서 만약, module이 Conv2d라면, dict의 key에 conv라는 문자열이 포함되지 않을 때까지 dict를 계속 앞에서부터 제거하며
    # split('.')[-1]로 가장 마지막 것(bias, weight) 등을 찾은 다음 module[att] = weight를 주어 파라미터를 업데이트 시킨다.
    # 가장 마지막에 state_dict가 빈 배열이 되었는지를 확인하여 마무리.
    flag = False
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            if not flag:
                key, value = state_dict.popitem(last=False)
                flag = False

            point = '.'.join(key.split('.')[:-1])
            while point in key:
                if 'weight' in key or 'bias' in key:
                    setattr(module, key.split('.')[-1], nn.Parameter(value.float()))
                else:
                    setattr(module, key.split('.')[-1], value.float())
                print(module, key)
                
                if state_dict:
                    key, value = state_dict.popitem(last=False)
                    flag = True
                else:
                    break

    assert not state_dict

    torch.save(model.state_dict(), 'ENetB2_VggFace2_modified.pt')
    print('loaded')


def make_model_from_pretrained(weights_path):

    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)

    model = EfficientNet.from_pretrained('EfficientNet-b2')
    model._fc = nn.Linear(1408, 5).to(device) #last layer을 out에 맞게 바꿔줌

    flag = False
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
            if not flag:
                key, value = state_dict.popitem(last=False)
                flag = False

            point = '.'.join(key.split('.')[:-1])
            while point in key:
                if 'weight' in key or 'bias' in key:
                    setattr(module, key.split('.')[-1], nn.Parameter(value.float()))
                else:
                    setattr(module, key.split('.')[-1], value.float())

                if state_dict:
                    key, value = state_dict.popitem(last=False)
                    flag = True
                else:
                    break
    assert not state_dict
    return model
