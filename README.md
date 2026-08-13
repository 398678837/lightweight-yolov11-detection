# 轻量化YOLOv11红绿灯识别Demo

基于 Ultralytics YOLOv11 的红绿灯状态识别工具，支持红/黄/绿灯检测与危险预警，跨平台兼容。

## 功能特点

- 🚦 **红绿灯状态识别**：精准识别红灯、黄灯、绿灯三种状态
- ⚠️ **危险预警**：自动检测危险信号（红灯/黄灯），输出警告信息
- 🚀 **极简代码**：仅图片批量推理，不含训练/视频模块
- 🪶 **轻量化模型**：默认使用预训练交通灯模型，快速推理
- ⚙️ **参数可配**：YAML配置 + 命令行覆盖
- 📁 **自动管理**：自动创建输出目录，自动保存结果
- 🖥️ **跨平台**：兼容 Windows / macOS / Linux

## 项目结构

```
lightweight-yolov11-detection/
├── main.py              # 主程序（批量图片推理 + 状态识别）
├── config.yaml          # 配置文件（模型/参数/红绿灯专有配置）
├── requirements.txt     # 依赖列表
├── README.md            # 本文档
├── weights/
│   └── traffic_light_best.pt  # 交通灯预训练权重（红/黄/绿三类）
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

将待检测的交通场景图片放入 `inputs/` 目录，支持格式：
`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.webp`

### 4. 运行检测

```bash
# 默认配置（红绿灯检测）
python main.py

# 使用GPU加速
python main.py --device 0

# 调整置信度阈值
python main.py --conf 0.3

# 自定义输入输出目录
python main.py -i ./my_images -o ./my_results

# 使用其他通用模型（非红绿灯）
python main.py -m yolo11n

# 使用自定义权重
python main.py -m weights/my_custom_model.pt
```

### 5. 查看结果

检测结果自动保存到 `outputs/` 目录：
- `*_result.jpg` — 标注检测框的可视化图片（红框=红灯，黄框=黄灯，绿框=绿灯）
- `*_result.json` — 单张图片检测详情（含中文状态和危险标记）
- `summary.json` — 本次运行汇总报告（含交通灯统计和危险预警）

## 红绿灯检测配置

编辑 `config.yaml` 中的 `traffic_light` 部分：

```yaml
traffic_light:
  enabled: true
  class_map:
    0: "绿灯"
    1: "红灯"
    2: "黄灯"
  colors:
    green: [0, 255, 0]
    red: [0, 0, 255]
    yellow: [0, 255, 255]
  danger_states:
    - "红灯"
    - "黄灯"
```

### 状态说明

| 状态 | 颜色 | 含义 | 危险等级 |
|------|------|------|----------|
| 绿灯 | 🟢 绿色 | 允许通行 | 安全 |
| 黄灯 | 🟡 黄色 | 警示/即将变灯 | ⚠️ 注意 |
| 红灯 | 🔴 红色 | 禁止通行 | ⚠️ 危险 |

## 模型说明

当前使用的预训练模型基于 Roboflow 交通灯数据集训练，支持 3 类目标：

| 类别 | 说明 |
|------|------|
| green | 绿灯 |
| red | 红灯 |
| yellow | 黄灯 |

### 切换到通用目标检测

如果需要检测其他目标（非红绿灯），修改 `config.yaml`：

```yaml
model:
  path: ""
  name: "yolo11n"
traffic_light:
  enabled: false
```

## 常见问题

**Q: 模型检测不准怎么办？**
A: 建议使用更清晰、光照条件好的图片，或在 config.yaml 中调高 `conf_threshold`（默认0.25）。

**Q: 如何使用自己的交通灯模型？**
A: 将权重文件放入 `weights/` 目录，在 config.yaml 中设置 `model.path: "weights/your_model.pt"`。

**Q: 如何使用GPU加速？**
A: 安装对应CUDA版本的PyTorch，设置 `device: "0"`。

**Q: 支持子目录扫描吗？**
A: 支持，`inputs/` 下的子目录中的图片也会被递归检测。

## 许可证

MIT