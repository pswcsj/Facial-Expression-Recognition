import torch.optim as optim
from tqdm import tqdm
from model.robust_optimization import RobustOptimizer
from model.loss import robust_loss


def train(model, epochs, lr, train_dataloader, test_dataloader):
    total_train_loss, train_correct, train_total = 0, 0, 0
    optimizer = RobustOptimizer(filter(lambda p: p.requires_grad, model.parameters()), optim.Adam, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            first_loss = robust_loss(model(images), labels)
            first_loss.backward()  # first_loss에 대한 weight의 gradient를 계산하여 model weight에 저장
            optimizer.first_step(zero_grad=True)

            # 중간 파라미터로 train_loss 계산
            train_loss = robust_loss(model(images), labels)
            train_loss.backward()
            optimizer.second_step(zero_grad=True)

            total_train_loss = total_train_loss + train_loss
            pred = output.max(1, keepdim=True)[1]
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
        torch.save(f'model/pretrained/emotion/{model.state_dict()}', f'{epoch}model.pt')
        with torch.no_grad():
            model.eval()
            test_loss, correct, test_total = 0, 0, 0
            for i, (images, labels) in enumerate(test_dataloader):
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                test_loss += module_loss.robust_loss(output, labels).item()

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

                test_total += labels.size(0)
        print(
            f"Epoch{epoch}/{epochs} - loss : {total_train_loss:.4f} - acc: {100. * train_correct / train_total:.4f} - test_loss : {test_loss:.4f} - test_acc: {100. * correct / test_total:.4f}"
        )
        scheduler.step()
