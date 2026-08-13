"""
轻量化YOLOv11红绿灯识别Demo - 批量图片推理
支持交通灯状态识别（红灯/黄灯/绿灯）与危险预警
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
    return sorted(files, key=lambda p: p.name)


def get_traffic_light_style(tl_cfg: dict, cls_name: str) -> tuple:
    """根据交通灯状态获取标注颜色和中文状态"""
    class_map = tl_cfg.get("class_map", {})
    colors = tl_cfg.get("colors", {})

    cn_state = None
    for cid, cname in class_map.items():
        if class_map.get(cid) and cls_name.lower() == cname.lower():
            cn_state = cname
            break

    if cls_name.lower() == "red":
        color = tuple(colors.get("red", [0, 0, 255]))
        cn_state = cn_state or "红灯"
    elif cls_name.lower() == "yellow":
        color = tuple(colors.get("yellow", [0, 255, 255]))
        cn_state = cn_state or "黄灯"
    elif cls_name.lower() == "green":
        color = tuple(colors.get("green", [0, 255, 0]))
        cn_state = cn_state or "绿灯"
    else:
        color = (255, 255, 255)
        cn_state = cn_state or cls_name

    return color, cn_state


def run_inference(config: dict) -> None:
    """执行批量图片推理（红绿灯检测模式）"""
    model_cfg = config["model"]
    infer_cfg = config["inference"]
    paths_cfg = config["paths"]
    tl_cfg = config.get("traffic_light", {})

    model_path = model_cfg.get("path") or model_cfg.get("name", "yolo11n")

    base_dir = Path(__file__).parent
    input_dir = base_dir / paths_cfg["input_dir"]
    output_dir = base_dir / paths_cfg["output_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 加载模型: {model_path}")
    model = YOLO(model_path)

    class_names = list(model.names.values())
    is_traffic_light = tl_cfg.get("enabled", False) and any(
        c in class_names for c in ["red", "green", "yellow"]
    )

    if is_traffic_light:
        print(f"[INFO] 交通灯检测模式已启用")
        print(f"[INFO] 模型类别: {class_names}")
    else:
        print(f"[INFO] 通用目标检测模式")

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

    danger_states = tl_cfg.get("danger_states", [])

    results_list = []
    traffic_light_stats = {"red": 0, "yellow": 0, "green": 0, "other": 0}

    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] 检测: {img_path.name}", end=" ... ")

        results = model(
            source=str(img_path),
            conf=infer_cfg["conf_threshold"],
            iou=infer_cfg["iou_threshold"],
            imgsz=infer_cfg["imgsz"],
            device=infer_cfg["device"],
            save=False,
            verbose=False,
        )

        result = results[0]
        num_boxes = len(result.boxes)

        detections = []
        danger_count = 0

        if num_boxes > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                cn_state = cls_name
                if is_traffic_light:
                    _, cn_state = get_traffic_light_style(tl_cfg, cls_name)
                    if cn_state in danger_states:
                        danger_count += 1
                    state_key = cls_name.lower()
                    if state_key in traffic_light_stats:
                        traffic_light_stats[state_key] += 1
                    else:
                        traffic_light_stats["other"] += 1

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "state_cn": cn_state,
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 2) for v in xyxy],
                    "is_danger": cn_state in danger_states if is_traffic_light else False,
                })

        state_summary = ""
        if is_traffic_light and num_boxes > 0:
            states = [d["state_cn"] for d in detections]
            state_summary = f" | 状态: {', '.join(states)}"
            if danger_count > 0:
                state_summary += f" ⚠️ 危险({danger_count})"

        print(f"发现 {num_boxes} 个目标{state_summary}")

        if infer_cfg["save_visualization"]:
            from PIL import Image as PILImage, ImageDraw, ImageFont

            annotated = result.plot()
            pil_img = PILImage.fromarray(annotated)

            if is_traffic_light and num_boxes > 0:
                draw = ImageDraw.Draw(pil_img)
                font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",
                    "/System/Library/Fonts/STHeiti Medium.ttc",
                    "/System/Library/Fonts/Hiragino Sans GB.ttc",
                    "C:/Windows/Fonts/msyh.ttc",
                    "C:/Windows/Fonts/simhei.ttf",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                ]
                font = None
                for fp in font_paths:
                    try:
                        font = ImageFont.truetype(fp, 18)
                        break
                    except (IOError, OSError):
                        continue
                if font is None:
                    font = ImageFont.load_default()

                for det in detections:
                    cls_name = det["class_name"]
                    color, cn_state = get_traffic_light_style(tl_cfg, cls_name)
                    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                    label = f"{cn_state} {det['confidence']:.2f}"

                    bbox = draw.textbbox((0, 0), label, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.rectangle(
                        [x1, y1 - th - 8, x1 + tw + 6, y1],
                        fill=color
                    )
                    draw.text((x1 + 3, y1 - th - 6), label, fill=(255, 255, 255), font=font)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            out_path = output_dir / f"{img_path.stem}_result{img_path.suffix}"
            pil_img.save(str(out_path))

        if infer_cfg["save_json"]:
            json_path = output_dir / f"{img_path.stem}_result.json"
            result_data = {
                "image": img_path.name,
                "total_detections": num_boxes,
                "is_traffic_light": is_traffic_light,
                "detections": detections,
                "danger_count": danger_count,
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

        results_list.append({
            "image": img_path.name,
            "num_detections": num_boxes,
            "danger_count": danger_count,
            "detections": detections,
        })

    print("-" * 50)
    print(f"[完成] 共处理 {len(image_files)} 张图片")
    total_detections = sum(r["num_detections"] for r in results_list)
    total_danger = sum(r["danger_count"] for r in results_list)
    print(f"[完成] 共检测到 {total_detections} 个目标")

    if is_traffic_light:
        print(f"[统计] 红灯: {traffic_light_stats['red']} | 黄灯: {traffic_light_stats['yellow']} | 绿灯: {traffic_light_stats['green']}")
        if total_danger > 0:
            print(f"[警告] ⚠️  检测到 {total_danger} 个危险信号（红灯/黄灯），请注意行车安全！")
        else:
            print(f"[安全] 当前未检测到危险信号")

    print(f"[完成] 结果保存在: {output_dir}")

    summary_path = output_dir / "summary.json"
    summary_data = {
        "total_images": len(image_files),
        "total_detections": total_detections,
        "total_danger_count": total_danger,
        "model": model_path,
        "device": infer_cfg["device"],
        "conf_threshold": infer_cfg["conf_threshold"],
        "is_traffic_light": is_traffic_light,
        "results": results_list,
    }
    if is_traffic_light:
        summary_data["traffic_light_stats"] = traffic_light_stats

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"[完成] 汇总报告: {summary_path}")


def main():
    """主入口：解析命令行参数并运行推理"""
    parser = argparse.ArgumentParser(
        description="轻量化YOLOv11红绿灯识别Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                                    # 使用默认配置（红绿灯检测）
  python main.py --conf 0.5 --imgsz 1280            # 覆盖参数
  python main.py --model yolo11s --device 0         # 使用GPU和更大模型
  python main.py --input ./my_images --output ./my_results
  python main.py -m weights/traffic_light_best.pt   # 使用自定义红绿灯模型
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
        help="模型名称或路径 (如 yolo11n, yolo11s, 或 weights/best.pt)",
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

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))

    if args.model:
        config["model"]["path"] = args.model
        config["model"]["name"] = ""
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

    print("=" * 50)
    print("  轻量化YOLOv11红绿灯识别Demo")
    print("=" * 50)

    run_inference(config)


if __name__ == "__main__":
    main()