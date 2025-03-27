'''
Author: matiastang
Date: 2025-03-26 18:00:51
LastEditors: matiastang
LastEditTime: 2025-03-27 11:05:37
FilePath: /pytorch-learning/src/mnist_test.py
Description: 加载本地的手写图片，用训练好的模型进行预测
'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 转换操作：将图像转换为 tensor，并归一化
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.RandomInvert(p=1.0),  # 颜色反转，确保 100% 反转（黑变白，白变黑）
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 加载本地数据集
test_data = datasets.ImageFolder(root='./data/MNIST_TEST', transform=transform)

# 创建数据加载器
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层 → 隐藏层1（128个神经元）
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1 → 隐藏层2（64个神经元）
        self.fc3 = nn.Linear(64, 10)  # 隐藏层2 → 输出层（10类数字）
        self.relu = nn.ReLU()  # ReLU 激活函数 避免梯度消失问题，提高模型性能

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将 28x28 的图像展平为 1D 向量
        x = self.relu(self.fc1(x))  # 经过第一层并激活
        x = self.relu(self.fc2(x))  # 经过第二层并激活
        x = self.fc3(x)  # 经过输出层，不加激活（CrossEntropyLoss 已经包含 softmax）
        return x


# 加载保存的模型
model = Net()  # 创建模型实例，这里需要用到我们定义的神经网络模型
model.load_state_dict(torch.load("models/mnist_model.pth"))
model.eval()  # 将模型设置为评估模式，禁用 dropout 和 batch normalization 等训练时使用的层。


# 推理
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images, labels = next(iter(test_loader))  # 获取测试集的第一批数据
images, labels = images.to(device), labels.to(device)  # 移动到 GPU 或 CPU 上
outputs = model(images)  # 前向传播
_, predicted = torch.max(outputs, 1)  # 获取预测结果

# 可视化预测结果
fig, axes = plt.subplots(1, 10, figsize=(12, 4))  # 创建一个 1x6 的子图
for i in range(10):
    img = images[i].cpu().numpy().squeeze()  # 将张量转换为 NumPy 数组，并去除额外的维度
    axes[i].imshow(img, cmap='gray')  # 显示灰度图像
    axes[i].set_title(f'Pred: {predicted[i].item()}')  # 显示预测结果
    axes[i].axis('off')  # 隐藏坐标轴
plt.show()  # 显示图像
