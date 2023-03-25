import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from model.EfficientNet import EfficientNet
import model.loss as module_loss
from data_loader import AffectNetDataLoader
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from utils.util import load_pretrained
import timm
from trainer.trainer import train
from model.robust_optimization import RobustOptimizer
parser = argparse.ArgumentParser()  #

a = {'0': 'neutral', '1': 'happy', '2': 'sad', '3': 'surprise', '4': 'anger'}  # 라벨링
# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--path', default="./dataset")

args = parser.parse_args()

eps = 0.05
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
path = args.path
batch_size = args.batch
first_epochs = 3
second_epochs = 5

if __name__ == '__main__':
    train_dataloader = AffectNetDataLoader(path=path, batch_size=batch_size, train=True)  # 학습용 데이터셋
    test_dataloader = AffectNetDataLoader(path=path, batch_size=batch_size, train=False, shuffle=False)  # 테스트용 데이터셋
    # 모델 정의한 후 device로 보내기
    model = timm.create_model('tf_efficientnet_b2_ns', pretrained=False)
    model.classifier = torch.nn.Identity()
    # model.load_state_dict(torch.load('model/pretrained/face_recognition/ENetB2_VggFace2.pt', map_location=torch.device('cpu')))
    model.load_state_dict(torch.load('model/pretrained/face_recognition/ENetB2_VggFace2.pt'))
    model.classifier = nn.Linear(in_features=1408, out_features=5)
    model = model.to(device)
    print(model)

    #마지막 레이어 빼고 freeze
    for param in model.parameters():
        param.requires_grad = False
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True
    
    train(model, first_epochs, 3e-5, train_dataloader, test_dataloader)
    for param in model.parameters():
        param.requires_grad = True

    del train_dataloader
    train_dataloader = AffectNetDataLoader(path=path, batch_size=64, train=True)  # 학습용 데이터셋
    train(model, second_epochs, 3e-5, train_dataloader, test_dataloader)
    # plt.plot(test_losses, label="test_loss")
    # plt.plot(train_losses, label="train_loss")
    # plt.legend()
    # plt.show()

