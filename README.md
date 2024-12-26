# 基于MHFormer的健身助手[UCAS HCI 2024]

![zhu](figure/zhu.gif)


## 安装

- 创建 Conda 环境：`conda create -n mhformer python=3.9`
- 按照 [官方说明](https://pytorch.org/) 安装 PyTorch 1.7.1 和 Torchvision 0.8.2
- `pip3 install -r requirements.txt`

## 下载预训练模型

预训练模型可以在 [此处](https://drive.google.com/drive/folders/1UWuaJ_nE19x2aM-Th221UpdhRPSCFwZa?usp=sharing) 找到，下载后请放到 `./checkpoint/pretrained` 目录。



## 姿态识别

### ui界面

运行下列代码

```sh
python ui/gradio_estimate.py
```

## 模型训练

打开ui/train.py，修改第54行，将需训练的姿态动作视频放入一个文件夹当中，并修改训练路径，同时修改mode，接下来会自动训练该文件夹下的所有视频。

```python
# 调用实例，就像调用函数一样   
    file_path = './ui/train/{PoseYouWantToTrain}'
    mode = '{PoseYouWantToTrain}' # 中文
```

```bash
${POSE_ROOT}/
|-- ui
|   |-- dataset
|   |   |-- {PoseYouWantToTrain}
|   |   |   |-- vedio1.mp4
|   |   |   |-- vedio2.mp4
```

注意训练前请先运行一下相关动作的_data文件，但切记，开始训练后不要再运行该文件，该文件会刷新训练后的npz文件，同样对于新的训练集，直接替换掉原来的训练集，开始训练即可。



如果从未训练，首先运行：
```sh
python python ui/{PoseYouWantToTrain}_data.py
```

然后运行:
```sh
python ui/train.py
```

