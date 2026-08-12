"""
轻量化YOLOv11图片检测Demo - 批量图片推理
支持命令行参数覆盖config.yaml配置
"""
import argparse
import json
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """加载YAML配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_image_files(input_dir: Path, extensions: list) -> list:
    """获取目录下所有支持的图片文件（递归）"""
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))
    # 按文件名排序，保证处理顺序一致
    return sorted(files, key=lambda p: p.name)


def run_inference(config: dict) -> None:
    """执行批量图片推理"""
    # ---- 解析配置 ----
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    paths_cfg = config["paths"]

    # 模型路径：优先用自定义路径，否则用预训练名称
    model_path = model_cfg.get("path") or model_cfg.get("name", "yolov11n")

    # 路径处理（兼容Win/Mac）
    base_dir = Path(__file__).parent
    input_dir = base_dir / paths_cfg["input_dir"]
    output_dir = base_dir / paths_cfg["output_dir"]

    # 自动创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    # ---- 收集待检测图片 ----
    extensions = paths_cfg.get("image_extensions", [".jpg", ".jpeg", ".png"])
    image_files = get_image_files(input_dir, extensions)

    if not image_files:
        print(f"[WARN] 输入目录为空: {input_dir}")
        print("       请将图片放入 inputs/ 目录后重试")
        return

    print(f"[INFO] 待检测图片: {len(image_files)} 张")
    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")
    print("-" * 50)

    # ---- 批量推理 ----
    results_list = []
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 检测: {img_path.name}", end=" ... ")

        # 执行推理
        results = model(
            source=str(img_path),
            conf=infer_cfg["conf_threshold"],
            iou=infer_cfg["iou_threshold"],
            imgsz=infer_cfg["imgsz"],
            device=infer_cfg["device"],
            save=False,  # 手动控制保存
            verbose=False,
        )

        result = results[0]

        # 统计检测结果
        num_boxes = len(result.boxes)
        print(f"发现 {num_boxes} 个目标")

        # 收集检测详情
        detections = []
        if num_boxes > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 2) for v in xyxy],
                })

        # 保存可视化标注图
        if infer_cfg["save_visualization"]:
            annotated = result.plot()  # 绘制检测框
            from PIL import Image
            out_path = output_dir / f"{img_path.stem}_result{img_path.suffix}"
            Image.fromarray(annotated).save(str(out_path))

        # 保存JSON结果
        if infer_cfg["save_json"]:
            json_path = output_dir / f"{img_path.stem}_result.json"
            result_data = {
                "image": img_path.name,
                "total_detections": num_boxes,
                "detections": detections,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

        results_list.append({
            "image": img_path.name,
            "num_detections": num_boxes,
            "detections": detections,
        })

    # ---- 汇总报告 ----
    print("-" * 50)
    print(f"[完成] 共处理 {len(image_files)} 张图片")
    total_detections = sum(r["num_detections"] for r in results_list)
    print(f"[完成] 共检测到 {total_detections} 个目标")
    print(f"[完成] 结果保存在: {output_dir}")

    # 保存汇总JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_images": len(image_files),
            "total_detections": total_detections,
            "model": model_path,
            "device": infer_cfg["device"],
            "conf_threshold": infer_cfg["conf_threshold"],
            "results": results_list,
        }, f, ensure_ascii=False, indent=2)
    print(f"[完成] 汇总报告: {summary_path}")


def main():
    """主入口：解析命令行参数并运行推理"""
    parser = argparse.ArgumentParser(
        description="轻量化YOLOv11图片批量检测Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 使用默认配置
  python main.py --conf 0.5 --imgsz 1280            # 覆盖参数
  python main.py --model yolov11s --device 0        # 使用GPU和更大模型
  python main.py --input ./my_images --output ./my_results
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="模型名称或路径 (如 yolov11n, yolov11s, 或 weights/best.pt)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="输入图片目录",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出结果目录",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="置信度阈值 (0-1)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU阈值 (0-1)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="输入图片尺寸",
    )
    parser.add_argument(
        "--device", "-d",
        default=None,
        help="推理设备 (cpu / 0 / 0,1)",
    )
    parser.add_argument(
        "--no-save-vis",
        action="store_true",
        help="不保存可视化标注图",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="不保存检测结果JSON",
    )

    args = parser.parse_args()

    # ---- 加载配置 ----
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    # ---- 命令行覆盖配置 ----
    if args.model:
        config["model"]["name"] = args.model
        config["model"]["path"] = ""  # 清空自定义路径
    if args.input:
        config["paths"]["input_dir"] = args.input
    if args.output:
        config["paths"]["output_dir"] = args.output
    if args.conf is not None:
        config["inference"]["conf_threshold"] = args.conf
    if args.iou is not None:
        config["inference"]["iou_threshold"] = args.iou
    if args.imgsz is not None:
        config["inference"]["imgsz"] = args.imgsz
    if args.device:
        config["inference"]["device"] = args.device
    if args.no_save_vis:
        config["inference"]["save_visualization"] = False
    if args.no_save_json:
        config["inference"]["save_json"] = False

    # ---- 打印当前配置 ----
    print("=" * 50)
    print("  轻量化YOLOv11图片检测Demo")
    print("=" * 50)

    # ---- 运行推理 ----
    run_inference(config)


if __name__ == "__main__":
    main()