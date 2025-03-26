<!--
 * @Author: matiastang
 * @Date: 2025-03-25 17:09:10
 * @LastEditors: matiastang
 * @LastEditTime: 2025-03-26 16:42:41
 * @FilePath: /pytorch-learning/DEV.md
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

## 虚拟环境

通过`uv`创建虚拟环境
```sh
$ uv venv --python 3.12.9
Using CPython 3.12.9
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

进入虚拟环境
```sh
$ uv venv/bin/activate
(pytorch-learning)$  
```

`uv`初始化
```sh
$ uv init   
Initialized project `pytorch-learning`
```

由于在`VSCode`中将`Python`环境切换为刚刚创建的虚拟环境，这个环境里面没有`uv`，所以需要重新安装依赖
```sh
$ pip3 install uv
Collecting uv
  Downloading uv-0.6.9-py3-none-macosx_11_0_arm64.whl.metadata (11 kB)
Downloading uv-0.6.9-py3-none-macosx_11_0_arm64.whl (14.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.7/14.7 MB 1.4 MB/s eta 0:00:00
Installing collected packages: uv
Successfully installed uv-0.6.9
```

查看版本
```sh
$ uv --version
uv 0.6.9 (3d9460278 2025-03-20)
```

安装`pytorch`依赖，由于`uv`在安装过程中会在`~/.cache/uv/`路径下，创建缓存文件夹，所以需要使用`sudo`权限，不然会提示`Permission denied`。
```sh
$ sudo uv add torch torchvision
Resolved 28 packages in 3.74s
Prepared 13 packages in 1m 04s
Installed 13 packages in 403ms
 + filelock==3.18.0
 + fsspec==2025.3.0
 + jinja2==3.1.6
 + markupsafe==3.0.2
 + mpmath==1.3.0
 + networkx==3.4.2
 + numpy==2.2.4
 + pillow==11.1.0
 + setuptools==78.0.2
 + sympy==1.13.1
 + torch==2.6.0
 + torchvision==0.21.0
 + typing-extensions==4.12.2
```

查看依赖列表
```sh
$ uv pip list
Package           Version
----------------- --------
filelock          3.18.0
fsspec            2025.3.0
jinja2            3.1.6
markupsafe        3.0.2
mpmath            1.3.0
networkx          3.4.2
numpy             2.2.4
pillow            11.1.0
setuptools        78.0.2
sympy             1.13.1
torch             2.6.0
torchvision       0.21.0
typing-extensions 4.12.2
```

运行`main.py`
```sh
$ uv run main.py 
Hello from pytorch-learning!
```

安装`matplotlib`依赖
```sh
$ sudo uv add matplotlib
Password:
\░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0/
Resolved 37 packages in 2.02s
Prepared 9 packages in 8.41s
Installed 9 packages in 28ms
 + contourpy==1.3.1
 + cycler==0.12.1
 + fonttools==4.56.0
 + kiwisolver==1.4.8
 + matplotlib==3.10.1
 + packaging==24.2
 + pyparsing==3.2.3
 + python-dateutil==2.9.0.post0
 + six==1.17.0
```

运行手写数字识别测试
```sh
$ sudo uv run ./src/mnist.py
Password:
Matplotlib is building the font cache; this may take a moment.
100.0%
100.0%
100.0%
100.0%
Epoch 1, Loss: 0.3962
Epoch 2, Loss: 0.1945
Epoch 3, Loss: 0.1406
Epoch 4, Loss: 0.1137
Epoch 5, Loss: 0.0966
Accuracy: 96.08%
```

这里使用`sudo`来运行，是因为`mnist.py`文件没有运行权限，也可以使用`chmod +x ./src/mnist.py`修改权限后直接运行。

```sh
$ uv run ./src/mnist.py
Start training time: 2175937.410244125
Epoch 1, Loss: 0.4008
Epoch 2, Loss: 0.1916
Epoch 3, Loss: 0.1389
Epoch 4, Loss: 0.1131
Epoch 5, Loss: 0.0953
End training time: 2175952.914613375, diff time = 15.504369 seconds
Accuracy: 96.45%
```