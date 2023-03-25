from tqdm import tqdm


def train(model, epochs, train_dataloader, test_dataloader, optimizer):
    total_train_loss, train_correct, train_total = 0, 0, 0

    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_dataloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            first_loss = module_loss.robust_loss(model(images), labels)
            first_loss.backward()  # first_loss에 대한 weight의 gradient를 계산하여 model weight에 저장

            # gradient vector 계산
            grad_vector = []
            for parameter in model.classifier.parameters():
                grad_vector.append(torch.flatten(parameter.grad))
            # torch.norm : 벡터의 크기를 구해주는 함수.
            grad_vector = torch.cat(grad_vector)
            grad_vector = grad_vector / (torch.norm(grad_vector) + 1e-6)  # 단위 벡터로 만들어줌
            grad_vector = eps * grad_vector  # epsilon=0.05를 곱해줘 크기 조정

            # parameter를 벡터로 만듦
            theta = torch.nn.utils.parameters_to_vector(model.classifier.parameters())

            # 새로운 파라미터로 모델 파라미터 업데이트
            with torch.no_grad():  # gradient calculation을 안하는 명령어
                new_theta = theta + grad_vector
                nn.utils.vector_to_parameters(new_theta, model.classifier.parameters())

            # 중간 파라미터로 train_loss 계산
            train_loss = module_loss.robust_loss(model(images), labels)

            # 다시 원래 파라미터로 돌아간 후, backpropagation 진행(new_theta로 forward pass를 계산해야 하지만,
            # backward pass는 theta로 해야하기 때문
            nn.utils.vector_to_parameters(theta, model.classifier.parameters())
            train_loss.backward()
            optimizer.step()

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
