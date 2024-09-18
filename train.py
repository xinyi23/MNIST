import torch
import torch.optim as optim
import torch.nn as nn
from data_processing import load_data
from model import Net

def train_model(epochs=5, learning_rate=0.001):
    trainloader, _ = load_data()

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad() # 清除上一次梯度

            outputs = model(images) # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward() # 反向传播，计算梯度
            optimizer.step() # 更新模型参数

            total_loss += loss.item()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(trainloader):.4f}')

    torch.save(model, 'mnist_model.pth')


if __name__ == '__main__':
    train_model()