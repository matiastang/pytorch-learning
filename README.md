<!--
 * @Author: matiastang
 * @Date: 2025-03-25 17:09:10
 * @LastEditors: matiastang
 * @LastEditTime: 2025-03-27 17:28:16
 * @FilePath: /pytorch-learning/README.md
 * @Description: PyTorch Learning
-->
# PyTorch Learning

Pytorch学习项目

## 环境

### MacOS

* `MacOS Sequoia v15.1.1`
* `Python 3.12.9`
* `uv 0.6.9`

查看Python版本
```sh
$ python3 --version
Python 3.12.9
```

查看uv版本
```sh
$ uv --version
uv 0.6.9 (3d9460278 2025-03-20)
```

## 更新

### v0.0.4

- 添加**基于CIFAR10数据集的图形分类**
- Tensor和Autograd熟悉

### v0.0.3

- 将手写的`0-9`数字的图片加入训练集进行训练，然后使用训练的模型测试，与未加入训练集时的结果做对比。
- 添加**加载本地的手写图片，用训练好的模型进行预测**
- 添加`0-9`数字的图片，用于测试

### v0.0.2

- 添加**PyTorch 实现 MNIST 数据集手写数字识别**

### v0.0.1

- 项目基本配置