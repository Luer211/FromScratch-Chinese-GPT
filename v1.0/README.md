# FromScratch-Chinese-GPT v1.0

> 面向中文的字符级 GPT 模型训练框架，集成 Gradio 可视化界面，支持可视化调参，一站式训练与生成！

## 特性

- 字符级 GPT，从零构建自己的预训练模型
- Gradio 页面支持全流程：数据处理、自定义训练参数配置、模型训练、文本生成
- 轻量部署、低资源即可运行，适合教学、实验和中文创意写作任务

## 快速开始

本项目已在 [AutoDL 云平台](https://www.autodl.com/) 上成功运行测试，推荐使用 Conda 配置 GPU 加速环境。

AutoDL 云平台教程推荐：[开发人员如何微调大模型并暴露接口给后端调用（B站视频）](https://www.bilibili.com/video/BV1R6P7eVEtd)

### 1️、安装依赖（推荐方式 - Conda）

```bash
conda create -n FromScratch-Chinese-GPT python=3.10 -y
conda activate FromScratch-Chinese-GPT

# 安装 PyTorch + CUDA（适用于 RTX 系列，测试使用的是 RTX 3090 ）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其余依赖
pip install -r requirements.txt
```

### 2️、启动 Gradio 页面

```bash
python app.py
```

> 启动后访问：`http://localhost:7860` 进入交互界面
