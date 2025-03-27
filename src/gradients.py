'''
Author: matiastang
Date: 2025-03-27 16:21:07
LastEditors: matiastang
LastEditTime: 2025-03-27 16:32:12
FilePath: /pytorch-learning/src/gradients.py
Description: 自动微分，计算梯度
'''
import torch

# 自动微分

# 创建一个张量，设置 requires_grad=True 来跟踪与它相关的计算
x = torch.tensor(2.0, requires_grad=True)
print(x)
# tensor(2., requires_grad=True)

y = x ** 2
print(y)
# tensor(4., grad_fn=<PowBackward0>)

# y 作为操作的结果被创建，所以它有 grad_fn
print(y.grad_fn)

z = y * y * 3
out = z.mean()  # 对 z 求平均值
print(z, out)
# tensor(48., grad_fn=<MulBackward0>) tensor(48., grad_fn=<MeanBackward0>)

# .requires_grad_() 会改变张量的 requires_grad 标记
a = torch.randn(2, 2)
print(a.requires_grad)
# False

a.requires_grad_(True)
print(a.requires_grad)
# True

b = (a * a).sum()
print(b.grad_fn)
# <SumBackward0 object at 0x10a0e6980>

# 计算梯度
out.backward()
print(f'梯度为：{x.grad}')
# 梯度为：96.0


# 雅可比向量积
# 在某些情况下，我们希望计算向量值函数（具有多个输出的函数）的梯度。在这种情况下，我们希望计算雅可比向量积，而不是单独的梯度。

x = torch.randn(3, requires_grad=True)
y = x ** 2
while y.data.norm() < 1000:
    y = y * 2
    
print(y)
# tensor([1060.9614,  154.1167,  566.1297], grad_fn=<MulBackward0>)

# 现在在这种情况下，y 不再是一个标量。torch.autograd 不能够直接计算整个雅可比，但是如果我们只想要雅可比向量积，只需要简单的传递向量给 backward 作为参数。
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)
# tensor([ 2.9481e+02, -1.1236e+03, -2.1535e-01])

# 可以通过将代码包裹在 with torch.no_grad()，来停止对从跟踪历史中 的 .requires_grad=True 的张量自动求导。
print(x.requires_grad)
# True
print((x ** 2).requires_grad)
# True

with torch.no_grad():
    print((x ** 2).requires_grad)
    # False
