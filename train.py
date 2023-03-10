import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from model.EfficientNet import EfficientNet
from data_loader import FERTrainDataLoader, FERTestDataLoader, FERTestDataSet
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser()    #

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
parser.add_argument('--epochs', type=int, default=128)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = args.epochs
if __name__ == '__main__':
    train_dataloader = FERTrainDataLoader(batch_size=10)  # 학습용 데이터셋
    test_dataloader = FERTestDataLoader()  # 테스트용 데이터셋
    test_dataset = FERTestDataSet()
    # 모델 정의한 후 device로 보내기
    model = EfficientNet.from_pretrained('EfficientNet-b7', num_classes=7, in_channels=1).to(device)
    # loss function 정의 classification 문제이니 CrossEntropyLoss 사용
    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer로는 Adam 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    test_losses = []
    # running_loss = 0.0
    for epoch in range(epochs):
        model.train()
        print(f"{epoch}th epoch starting.")
        running_test_loss=0
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            train_loss = loss_function(model(images), labels)
            train_loss.backward()

            optimizer.step()

        model.eval()
        running_train_loss = 0.0
        running_test_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images, labels = images.to(device), labels.to(device)
            running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
        for i, (images, labels) in enumerate(test_dataloader, 0):
            images, labels = images.to(device), labels.to(device)
            running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
        train_losses.append(running_train_loss)
        test_losses.append(running_test_loss)

    test_loss, correct, total = 0, 0, 0
    for i, (images, labels) in enumerate(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        test_loss += loss_function(output, labels).item()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += labels.size(0)

    print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss / total, correct, total,
        100. * correct / total))
    plt.plot(test_losses, label="test_loss")
    plt.plot(train_losses, label="train_loss")
    plt.legend()
    plt.show()

    torch.save(model.state_dict(), 'model.pt')
