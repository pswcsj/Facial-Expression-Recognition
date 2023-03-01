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
parser = argparse.ArgumentParser()  #

a = {'0': 'neutral', '1': 'happy', '2': 'sad', '3': 'surprise', '4': 'anger'}  # 라벨링
# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--path', default="./dataset")

args = parser.parse_args()

eps = 0.05
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
path = args.path
if __name__ == '__main__':
    train_dataloader = AffectNetDataLoader(path=path, batch_size=128, train=True, num_workers=100)  # 학습용 데이터셋
    test_dataloader = AffectNetDataLoader(path=path, batch_size=2500, train=False, shuffle=False, num_workers=100)  # 테스트용 데이터셋

    # 모델 정의한 후 device로 보내기
    model = EfficientNet.from_pretrained('EfficientNet-b2').to(device)
    load_pretrained(model, './model/pretrained/face_recognition/ENetB2_VggFace2.pt')

    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_losses = []
    test_losses = []

    model.train()
    for epoch in range(epochs):
        print(f"{epoch}th epoch starting.")
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            first_loss = module_loss.robust_loss(model(images), labels)
            first_loss.backward()  # first_loss에 대한 weight의 gradient를 계산하여 model weight에 저장

            # gradient vector 계산
            grad_vector = []
            for parameter in model.parameters():
                grad_vector.append(torch.flatten(parameter.grad))
            grad_vector = torch.norm(torch.cat(grad_vector))  # 단위 벡터로 만들어줌
            grad_vector = eps * grad_vector  # epsilon=0.05를 곱해줘 크기 조정

            # parameter를 벡터로 만듦
            theta = torch.nn.utils.parameters_to_vector(model.parameters())

            # 새로운 파라미터로 모델 파라미터 업데이트
            with torch.no_grad():  # gradient calculation을 안하는 명령어
                new_theta = theta + grad_vector
                nn.utils.vector_to_parameters(new_theta, model.parameters())

            # 중간 파라미터로 train_loss 계산
            train_loss = module_loss.robust_loss(model(images), labels)

            # 다시 원래 파라미터로 돌아간 후, backpropagation 진행(new_theta로 forward pass를 계산해야 하지만,
            # backward pass는 theta로 해야하기 때문
            nn.utils.vector_to_parameters(theta, model.parameters())
            train_loss.backward()

            optimizer.step()
        scheduler.step()
        # model.eval()
        # running_train_loss = 0.0
        # running_test_loss = 0.0
        # for i, (images, labels) in enumerate(train_dataloader, 0):
        #     images, labels = images.to(device), labels.to(device)
        #     running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
        # for i, (images, labels) in enumerate(test_dataloader, 0):
        #     images, labels = images.to(device), labels.to(device)
        #     running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
        # train_losses.append(running_train_loss)
        # test_losses.append(running_test_loss)

    model.eval()
    test_loss, correct, total = 0, 0, 0
    for i, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        test_loss += module_loss.robust_loss(output, labels).item()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += labels.size(0)

    print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / total, correct, total,
        100. * correct / total))
    # plt.plot(test_losses, label="test_loss")
    # plt.plot(train_losses, label="train_loss")
    # plt.legend()
    # plt.show()

    torch.save(model.state_dict(), 'model.pt')
