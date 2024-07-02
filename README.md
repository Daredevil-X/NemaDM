# NemaAug: Few-shot Nematode Image Generation based on Diffusion Models and Morphological Constraints 

## 论文信息

- **标题**: NemaAug: Few-shot Nematode Image Generation based on Diffusion Models and Morphological Constraints
- **作者**: Xiong Ouyang, Jiayan Zhuang, Jianfeng Gu, Jiangjian Xiao, Weilun Ren, Sichao Ye, Ying Zhu
- **论文链接**: ---

## 环境安装

### 依赖工具

为了运行本项目，您需要安装以下：
- ![Stable-Diffusion-Webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) 
- ![Lora Fine-tuning](https://github.com/Akegarasu/lora-scripts) 
- ![ControlNet-Extension](https://github.com/Mikubill/sd-webui-controlnet)


### 约束生成

1. 克隆本仓库：
   ```bash
   git clone https://github.com/Daredevil-X/NemaDM
   cd NemaAug
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 项目结构

```
NemaAug/
├── README.md           # 本文件
├── requirements.txt    # 项目依赖库
├── pidinet
│   ├── load_params/    # 预训练权重
│   ├── model.py        # 模型定义
│   └── get_edges.py    # 获取边缘
├── deformation.py      # 形变算法
├── get_namatodes_edges.py  # 获取简单边缘
├── MLS_edges.py        # 批量形变
```

## 更新日志

### v1.0 (2024.07.02)
- 初始版本发布。
