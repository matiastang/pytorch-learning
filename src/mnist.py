'''
Author: matiastang
Date: 2025-03-26 15:27:16
LastEditors: matiastang
LastEditTime: 2025-03-27 14:31:35
FilePath: /pytorch-learning/src/mnist.py
Description: PyTorch 实现 MNIST 数据集手写数字识别
'''
import time  # time: 提供时间相关的函数，如获取当前时间、计算时间差等。
import torch  # PyTorch 的核心库，提供张量操作和自动微分功能。
import torch.nn as nn  # 用于构建神经网络的模块。
import torch.optim as optim  # 提供优化器，如 SGD、Adam 等
import torchvision  # torchvision: 处理计算机视觉数据的库，包含 MNIST 数据集。
# import torchvision.transforms as transforms  # torchvision.transforms: 提供数据预处理的工具，如归一化、数据增强等。
from torchvision import datasets, transforms
import matplotlib.pyplot as plt  # matplotlib.pyplot: 用于绘制图像和图表的库。
from torch.utils.data import ConcatDataset  # torch.utils.data: 提供数据集和数据加载器等工具。

# 1. 加载 MNIST 数据集

# 转换操作：将图像转换为 tensor，并归一化
custom_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.RandomInvert(p=1.0),  # 颜色反转，确保 100% 反转（黑变白，白变黑）
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 加载自定义数据集（ImageFolder 格式）
custom_set = datasets.ImageFolder(root='./data/MNIST_TEST', transform=custom_transform)

# 1.1 数据预处理设置
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为 PyTorch 张量，像素值范围变为 [0,1]。
    transforms.Normalize((0.5,), (0.5,)),  # 归一化数据，使其均值为 0，方差为 1，加快训练速度。
])
# 1.2.1 下载训练数据集
trainset = torchvision.datasets.MNIST(
    root='./data',  # 数据集的存储路径。
    train=True,  # train=True：加载训练集。
    download=True,  # download=True：如果数据集不存在，则从网上下载。
    transform=transform,  # transform：对数据集进行预处理。
)
# 1.2.2 加载训练数据集

# 合并两个数据集
trainset = ConcatDataset([custom_set, trainset])

trainloader = torch.utils.data.DataLoader(
    trainset,  # trainset：训练数据集。
    batch_size=64,  # batch_size=64：每次训练使用 64 张图片。
    shuffle=True,  # shuffle=True：随机打乱训练数据，避免过拟合。
)
# 1.3.1 下载测试数据集
testset = torchvision.datasets.MNIST(
    root='./data',  # 数据集的存储路径。
    train=False,  # train=False：加载测试集。
    download=True,  # download=True：如果数据集不存在，则从网上下载。
    transform=transform,  # transform：对数据集进行预处理。
)
# 1.3.2 加载测试数据集

# 合并两个数据集
testset = ConcatDataset([custom_set, testset])

testloader = torch.utils.data.DataLoader(
    testset,  # testset：测试数据集。
    batch_size=64,  # batch_size=64：每次测试使用 64 张图片。
    shuffle=False,  # shuffle=False：不随机打乱测试数据，保证测试结果的稳定性。
)


# 2. 定义神经网络模型
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


# 3. 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测是否有 GPU（CUDA），如果有，则在 GPU 运行，提高训练速度。
model = Net().to(device)  # 将模型加载到 GPU 或 CPU 上
criterion = nn.CrossEntropyLoss()  # 交叉熵损失：用于分类任务，适用于多类分类问题。
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器：比传统 SGD 更快收敛，lr=0.001 是学习率。

start_time = time.perf_counter()
print(f'Start training time: {start_time}')
num_epochs = 7  # 训练 5 个 epoch
for epoch in range(num_epochs):
    running_loss = 0.0  # 在每个 epoch 开始时，将损失初始化为 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)  # 移动到 GPU 或 CPU 上
        optimizer.zero_grad()  # 梯度清零，避免梯度累积，防止梯度累积
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累计损失
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")  # 打印每个 epoch 的平均损失

end_time = time.perf_counter()
print(f'End training time: {end_time}, diff time = {end_time - start_time:.6f} seconds')

# 4. 测试模型
correct = 0  # 初始化正确预测的数量
total = 0  # 初始化总测试样本数量
with torch.no_grad():  # 在测试阶段，不需要计算梯度，节省内存和计算资源。
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)  # 移动到 GPU 或 CPU 上
        outputs = model(images)  # 前向传播
        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        total += labels.size(0)  # 累计总测试样本数量
        correct += (predicted == labels).sum().item()  # 累计正确预测的数量

print(f'Accuracy: {100 * correct / total:.2f}%')  # 打印测试集上的准确率

# 5. 保存模型 & 加载模型

# 保存模型参数，会将 **模型参数** 保存到当前 工作目录 下的 `models/mnist_model.pth` 文件。
torch.save(model.state_dict(), 'models/mnist_model.pth')

# 如果加载到 GPU 训练的模型，但要在 CPU 上使用：model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))

# 加载保存的模型
mnist_model = Net()  # 创建模型实例，这里需要用到我们定义的神经网络模型
mnist_model.load_state_dict(torch.load("models/mnist_model.pth"))
mnist_model.eval()  # 将模型设置为评估模式，禁用 dropout 和 batch normalization 等训练时使用的层。

# 6. 可视化预测
images, labels = next(iter(testloader))  # 获取测试集的第一批数据
images, labels = images.to(device), labels.to(device)  # 移动到 GPU 或 CPU 上
outputs = mnist_model(images)  # 前向传播
_, predicted = torch.max(outputs, 1)  # 获取预测结果

# 可视化预测结果
fig, axes = plt.subplots(1, 10, figsize=(12, 4))  # 创建一个 1x6 的子图
for i in range(10):
    img = images[i].cpu().numpy().squeeze()  # 将张量转换为 NumPy 数组，并去除额外的维度
    axes[i].imshow(img, cmap='gray')  # 显示灰度图像
    axes[i].set_title(f'Pred: {predicted[i].item()}')  # 显示预测结果
    axes[i].axis('off')  # 隐藏坐标轴
plt.show()  # 显示图像
