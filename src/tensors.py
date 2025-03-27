'''
Author: matiastang
Date: 2025-03-27 16:01:24
LastEditors: matiastang
LastEditTime: 2025-03-27 16:14:10
FilePath: /pytorch-learning/src/tensors.py
Description: Tensors (张量) 学习
'''
import torch

# 创建

# 创建一个未初始化的5x3矩阵
x = torch.empty(5, 3)
print('======创建一个未初始化的5x3矩阵======')
print(x)

# 创建一个随机初始化的5x3矩阵
x = torch.rand(5, 3)
print('======创建一个随机初始化的5x3矩阵======')
print(x)

# 创建一个全0的5x3矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print('======创创建一个全0的5x3矩阵======')
print(x)

# 创建一个张量，直接从数据中创建
x = torch.tensor([5.5, 3])
print('======创建一个张量，直接从数据中创建======')
print(x)

# 通过已有的张量创建新的张量
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print('======结果是x两倍======')
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print('======结果与x相同，但是数据类型为float======')
print(x)                                      # 结果与x相同，但是数据类型为float

# 获取信息

# 获取维度信息
print('======获取维度信息======')
print(x.size())
# torch.Size是一个元组，所以它支持左右的元组操作。

# 操作

# 加法
y = torch.rand(5, 3)
print('======加法======')

print(x + y)

print(torch.add(x, y))

# 加法，指定输出
result = torch.empty(5, 3)
torch.add(x, y, out=result)

# in-place操作
y.add_(x)
print(y)
# 注意：任何以_结尾的操作都是in-place操作，即直接修改自身的数据。**任何使张量会发生变化的操作都有一个前缀 '_'**

# 改变大小
print('======改变tensor尺寸======')

# 如果想要改变一个张量的大小，可以使用torch.view()
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # -1表示该位置的大小由其他维度的大小推断出来

# item获取tensor元素

x = torch.randn(1)
print('======item获取tensor元素======')
print(x.item())
