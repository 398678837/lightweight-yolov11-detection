# 轻量化YOLOv11图片检测Demo

基于 Ultralytics YOLOv11 的极简图片批量推理工具，支持 Windows / macOS / Linux。

## 功能特点

- 🚀 **极简代码**：仅图片批量推理，不含训练/视频模块
- 🪶 **轻量化模型**：默认使用 YOLOv11n（Nano），快速轻量
- ⚙️ **参数可配**：YAML配置 + 命令行覆盖
- 📁 **自动管理**：自动创建输出目录，自动保存结果
- 🖥️ **跨平台**：兼容 Windows / macOS / Linux

## 项目结构

```
lightweight-yolov11-detection/
├── main.py              # 主程序（批量图片推理）
├── config.yaml          # 配置文件（模型/参数/路径）
├── requirements.txt     # 依赖列表
├── README.md            # 本文档
├── inputs/              # 待检测图片（用户放入）
└── outputs/             # 检测结果（自动创建）
```

## 快速开始

### 1. 环境要求

- Python >= 3.8
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 放入图片

将待检测的图片放入 `inputs/` 目录，支持格式：
`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp`

### 4. 运行检测

```bash
# 默认配置运行
python main.py

# 指定配置文件
python main.py --config config.yaml

# 命令行覆盖参数
python main.py --conf 0.5 --imgsz 1280

# 使用GPU加速
python main.py --device 0

# 使用更大的模型
python main.py --model yolov11s --device 0

# 自定义输入输出目录
python main.py -i ./my_images -o ./my_results

# 使用自定义权重
python main.py -m weights/best.pt
```

### 5. 查看结果

检测结果自动保存到 `outputs/` 目录：
- `*_result.jpg` — 标注检测框的可视化图片
- `*_result.json` — 单张图片检测详情
- `summary.json` — 本次运行汇总报告

## 配置说明

编辑 `config.yaml` 自定义参数：

```yaml
model:
  name: "yolov11n"          # 模型: yolov11n/s/m/l/x
  path: ""                   # 自定义权重路径（优先于name）

inference:
  conf_threshold: 0.25      # 置信度阈值
  iou_threshold: 0.45       # IoU阈值
  imgsz: 640                 # 输入尺寸
  device: "cpu"              # 设备: cpu / 0(GPU)
  save_visualization: true   # 保存标注图
  save_json: true            # 保存JSON结果

paths:
  input_dir: "inputs"        # 输入目录
  output_dir: "outputs"      # 输出目录
```

## 模型选择

| 模型 | 大小 | 速度 | 精度 | 适用场景 |
|------|------|------|------|----------|
| yolov11n | 5MB | ★★★★★ | ★★★ | 嵌入式/实时 |
| yolov11s | 18MB | ★★★★ | ★★★★ | 平衡选择 |
| yolov11m | 48MB | ★★★ | ★★★★ | 精度优先 |
| yolov11l | 72MB | ★★ | ★★★★★ | 高精度需求 |
| yolov11x | 99MB | ★ | ★★★★★ | 最高精度 |

## 常见问题

**Q: 首次运行很慢？**
A: 首次会自动下载模型权重（~5MB），之后会使用缓存。

**Q: 如何使用自己的模型？**
A: 将权重文件放入 `weights/` 目录，在 config.yaml 中设置 `model.path: "weights/best.pt"`。

**Q: 如何使用GPU加速？**
A: 安装对应CUDA版本的PyTorch，设置 `device: "0"`。

**Q: 支持子目录扫描吗？**
A: 支持，`inputs/` 下的子目录中的图片也会被递归检测。

## License

MIT