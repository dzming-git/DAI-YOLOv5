

# DAI-YOLOv5

DAI/基础设施层/目标检测/YOLOv5

## Docker 配置

### 下载cuda的runfile

使用的是11.7.1版本，操作系统是ubuntu20.04

https://developer.nvidia.com/cuda-11-7-1-download-archive

```Shell
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
```

### 挂载安装包文件夹

使用挂载的方式，在容器内部可以运行，配置文件的默认位置是 `～/DAI/pkgs`

将下载的runfile文件拷贝进这里

### 安装

```Shell
sh cuda_11.7.1_515.65.01_linux.run
```

**只安装 Cuda Toolkit**

### 设置环境变量

```Shell
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

source ~/.bashrc
```

### 验证

```Shell
nvidia-smi
nvcc --version
```