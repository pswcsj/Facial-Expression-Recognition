from data_loader import FERTrainDataLoader, FERTestDataLoader, FERTestDataSet
from model.EfficientNet import EfficientNet
model = EfficientNet.from_pretrained('EfficientNet-b7', num_classes=7, in_channels=1)

test_dataloader = FERTestDataLoader()  # 테스트용 데이터셋
train_dataloader = FERTrainDataLoader(batch_size=128)  # 학습용 데이터셋
model.train()
for epoch in range(10):
        print(f"{epoch}th epoch starting.")
        running_test_loss=0
        # for i, (images, labels) in enumerate(test_dataloader):
        #     images, labels = images.to(device), labels.to(device)
        #     print(images.shape, labels.shape)
        #     print(model(images).shape, labels.shape)
        #     running_test_loss += loss_function(model(images), labels).item() / images.shape[0]
        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images, labels
            optimizer.zero_grad()
            train_loss = loss_function(model(images), labels)
            train_loss.backward()

            optimizer.step()

        running_train_loss = 0.0
        running_test_loss = 0.0
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images, labels = images.to(device), labels.to(device)
            running_train_loss += loss_function(model(images), labels).item() / images.shape[0]
model.eval()
for i, (data, label) in enumerate(test_dataloader):
    model(data)
    print(model(data), label)
