import torch
from model import Net
from data_processing import load_data

def test_model():
    _, testloader = load_data()

    model = torch.load('mnist_model.pth')
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images) # 此时outputs是一个batch_size为64的张量
            _, predicted = torch.max(outputs, 1) # torch.max()返回最大值和最大值索引，此处丢弃最大值
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'准确率：{100*correct/total:.2f}%')

if __name__ == '__main__':
    test_model()