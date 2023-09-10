FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y git && \
    apt-get install -y libglib2.0-0 && \
    apt-get install -y python3.8 && \
    apt-get install -y python3-pip && \
    apt-get purge -y --auto-remove


COPY requirements_torch.txt workspace/requirements_torch.txt
RUN pip3 install -r workspace/requirements_torch.txt --index-url https://download.pytorch.org/whl/cu117

COPY requirements_yolo.txt workspace/requirements_yolo.txt
RUN pip3 install -r workspace/requirements_yolo.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements_others.txt workspace/requirements_others.txt
RUN pip3 install -r workspace/requirements_others.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# git 配置
RUN git config --global user.name "dzming" 
RUN git config --global user.email "dzm_work@163.com"
RUN git config --global http.proxy 127.0.0.1:7890
RUN git config --global https.proxy 127.0.0.1:7890

COPY . ./workspace
WORKDIR /workspace
