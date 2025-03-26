<!--
 * @Author: matiastang
 * @Date: 2025-03-25 17:09:10
 * @LastEditors: matiastang
 * @LastEditTime: 2025-03-26 10:14:04
 * @FilePath: /pytorch-learning/README.md
 * @Description: PyTorch Learning
-->
# PyTorch Learning

## 环境

使用`uv`来管理Python环境

```sh
$ uv --version
uv 0.6.8 (c1ef48276 2025-03-18)
```

```sh
$ uv venv --python 3.12.9
Using CPython 3.12.9
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
```

```sh
$ uv venv/bin/activate
(pytorch-learning)$  
```

```sh
$ uv init   
Initialized project `pytorch-learning`
```

由于在`VSCode`中切换了Python环境为刚刚创建的虚拟环境，这个环境里面没有`uv`，所以需要重新安装依赖
```sh
$ pip3 install uv
Collecting uv
  Downloading uv-0.6.9-py3-none-macosx_11_0_arm64.whl.metadata (11 kB)
Downloading uv-0.6.9-py3-none-macosx_11_0_arm64.whl (14.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14.7/14.7 MB 1.4 MB/s eta 0:00:00
Installing collected packages: uv
Successfully installed uv-0.6.9
```

```sh
$ uv --version
uv 0.6.9 (3d9460278 2025-03-20)
```

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

```sh
$ uv run main.py 
Hello from pytorch-learning!
```

## 更新

### v0.0.1

- 项目基本配置