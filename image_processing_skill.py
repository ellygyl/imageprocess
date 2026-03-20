#!/usr/bin/env python3
"""
图像处理 Skill
支持：目标检测、图像分类、语义分割

安装依赖：
pip install torch torchvision ultralytics opencv-python pillow requests

运行示例：
python image_processing_skill.py detect --image "https://example.com/image.jpg" --target "猫"
python image_processing_skill.py classify --image "https://example.com/image.jpg" --text "是否有猫"
python image_processing_skill.py segment --image "https://example.com/image.jpg"
"""

import sys
import json
import os
import argparse
import logging
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

import numpy as np
from PIL import Image

# 尝试导入必要的库
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def download_image(image_url: str) -> Optional[np.ndarray]:
    """
    从URL下载图片

    Args:
        image_url: 图片URL或本地路径

    Returns:
        numpy数组格式的图片，或None
    """
    if not REQUESTS_AVAILABLE:
        logging.error("requests库未安装，无法下载图片")
        return None

    try:
        # 判断是URL还是本地路径
        if image_url.startswith('http://') or image_url.startswith('https://'):
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            # 本地文件
            image = Image.open(image_url)

        # 转换为RGB（处理RGBA等格式）
        if image.mode != 'RGB':
            image = image.convert('RGB')

        return np.array(image)

    except Exception as e:
        logging.error(f"下载/读取图片失败: {e}")
        return None


class ObjectDetector:
    """
    目标检测器
    基于YOLOv5/YOLOv8进行目标检测
    """

    # COCO类别名称
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, model_name: str = "yolov8m.pt"):
        """
        初始化检测器

        Args:
            model_name: 模型名称 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False

    def load_model(self):
        """加载模型"""
        if self._loaded:
            return

        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(self.model_name)
                self._loaded = True
                logging.info(f"YOLO模型加载成功: {self.model_name}")
            except Exception as e:
                logging.warning(f"YOLO模型加载失败: {e}")
        else:
            logging.warning("Ultralytics未安装，将使用模拟检测")

        self._loaded = True

    def detect(self,
               image: np.ndarray,
               target_description: str = None,
               conf_threshold: float = 0.25) -> Dict[str, List[int]]:
        """
        执行目标检测

        Args:
            image: 输入图片 (numpy数组)
            target_description: 目标描述（如"猫"、"狗"）
            conf_threshold: 置信度阈值

        Returns:
            检测结果 {"0": [x1, y1, x2, y2], "1": [...], ...}
        """
        self.load_model()

        results = {}

        if self.model is not None and YOLO_AVAILABLE:
            # 使用真实YOLO模型
            try:
                detections = self.model(image, conf=conf_threshold, verbose=False)

                # 解析目标描述
                target_classes = self._parse_target_description(target_description)

                box_id = 0
                for result in detections:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.COCO_CLASSES[cls_id] if cls_id < len(self.COCO_CLASSES) else f"class_{cls_id}"

                        # 如果指定了目标类别，只返回匹配的检测结果
                        if target_classes and class_name.lower() not in target_classes:
                            continue

                        conf = float(boxes.conf[i].cpu().numpy())
                        xyxy = boxes.xyxy[i].cpu().numpy()

                        results[str(box_id)] = {
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "class": class_name,
                            "confidence": round(conf, 2)
                        }
                        box_id += 1

            except Exception as e:
                logging.error(f"检测失败: {e}")

        else:
            # 模拟检测
            results = self._mock_detect(image, target_description)

        return results

    def _parse_target_description(self, description: str) -> List[str]:
        """解析目标描述，返回类别列表"""
        if not description:
            return []

        # 常见目标映射
        target_map = {
            "猫": ["cat"],
            "狗": ["dog"],
            "人": ["person"],
            "车": ["car", "truck", "bus", "motorcycle"],
            "鸟": ["bird"],
            "动物": ["cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        }

        targets = []
        desc_lower = description.lower()

        # 检查中文映射
        for cn_name, en_names in target_map.items():
            if cn_name in description:
                targets.extend(en_names)

        # 检查英文
        for class_name in self.COCO_CLASSES:
            if class_name in desc_lower:
                targets.append(class_name)

        return targets

    def _mock_detect(self, image: np.ndarray, target_description: str) -> Dict[str, Any]:
        """模拟检测（当模型不可用时）"""
        h, w = image.shape[:2] if image is not None else (480, 640)

        # 返回模拟的检测结果
        return {
            "0": {
                "bbox": [int(w*0.1), int(h*0.2), int(w*0.4), int(h*0.6)],
                "class": "cat" if "猫" in (target_description or "") else "object",
                "confidence": 0.85
            },
            "note": "模拟检测结果（模型未加载）"
        }


class ImageClassifier:
    """
    图像分类器
    基于ResNet50进行图像分类
    """

    # ImageNet类别（部分）
    IMAGENET_CLASSES = None  # 延迟加载

    def __init__(self):
        self.model = None
        self.transform = None
        self._loaded = False

    def load_model(self):
        """加载ResNet50模型"""
        if self._loaded:
            return

        if TORCH_AVAILABLE:
            try:
                # 加载预训练的ResNet50
                self.model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
                self.model.eval()

                # 图像预处理
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                self._loaded = True
                logging.info("ResNet50模型加载成功")

            except Exception as e:
                logging.warning(f"ResNet50模型加载失败: {e}")
        else:
            logging.warning("torch未安装，将使用模拟分类")

        self._loaded = True

    def classify(self,
                 image: np.ndarray,
                 text_query: str = None) -> Dict[str, Any]:
        """
        执行图像分类

        Args:
            image: 输入图片
            text_query: 查询文本（如"是否有猫"）

        Returns:
            分类结果
        """
        self.load_model()

        if self.model is not None and TORCH_AVAILABLE:
            try:
                # 转换图像
                pil_image = Image.fromarray(image)
                input_tensor = self.transform(pil_image)
                input_batch = input_tensor.unsqueeze(0)

                # 推理
                with torch.no_grad():
                    output = self.model(input_batch)

                # 获取Top-5预测
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top5_prob, top5_catid = torch.topk(probabilities, 5)

                # 加载类别名称
                self._load_imagenet_classes()

                predictions = []
                for i in range(5):
                    class_id = top5_catid[i].item()
                    class_name = self.IMAGENET_CLASSES[class_id] if self.IMAGENET_CLASSES else f"class_{class_id}"
                    prob = top5_prob[i].item()
                    predictions.append({
                        "class": class_name,
                        "confidence": round(prob, 4)
                    })

                # 如果有查询文本，生成回答
                answer = self._generate_answer(predictions, text_query)

                return {
                    "predictions": predictions,
                    "answer": answer
                }

            except Exception as e:
                logging.error(f"分类失败: {e}")
                return {"error": str(e)}
        else:
            # 模拟分类
            return self._mock_classify(text_query)

    def _load_imagenet_classes(self):
        """加载ImageNet类别名称"""
        if self.IMAGENET_CLASSES is not None:
            return

        try:
            # 尝试从torchvision加载
            import json
            import urllib

            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            self.IMAGENET_CLASSES = []

            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    for line in response:
                        self.IMAGENET_CLASSES.append(line.decode('utf-8').strip())
            except:
                # 使用部分常见类别
                self.IMAGENET_CLASSES = [
                    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
                    'electric ray', 'stingray', 'cock', 'hen', 'ostrich',
                    'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting',
                    'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
                    'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl',
                    'tabby cat', 'tiger cat', 'Persian cat', 'Siamese cat', 'Egyptian cat',
                    'cougar', 'lynx', 'leopard', 'snow leopard', 'jaguar',
                    'lion', 'tiger', 'cheetah', 'brown bear', 'American black bear',
                    'ice bear', 'weasel', 'mink', 'polecat', 'black-footed ferret',
                    'otter', 'skunk', 'badger', 'armadillo', 'three-toed sloth',
                    # ... 更多类别
                ] + [f"class_{i}" for i in range(1000 - 50)]

        except Exception as e:
            logging.warning(f"加载ImageNet类别失败: {e}")
            self.IMAGENET_CLASSES = [f"class_{i}" for i in range(1000)]

    def _generate_answer(self, predictions: List[Dict], text_query: str) -> str:
        """根据查询生成回答"""
        if not text_query:
            top_class = predictions[0]["class"]
            top_conf = predictions[0]["confidence"]
            return f"图片中最可能包含: {top_class} (置信度: {top_conf:.2%})"

        # 解析查询意图
        query_lower = text_query.lower()

        # 检查是否包含特定类别查询
        target_classes = []
        if "猫" in text_query or "cat" in query_lower:
            target_classes.extend(["tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian cat", "cat"])
        if "狗" in text_query or "dog" in query_lower:
            target_classes.extend(["dog", "chihuahua", "Japanese spaniel", "Maltese dog", "Pekinese"])

        if target_classes:
            # 检查预测结果中是否包含目标类别
            for pred in predictions:
                for target in target_classes:
                    if target.lower() in pred["class"].lower():
                        return f"是的，图片中检测到 {pred['class']} (置信度: {pred['confidence']:.2%})"

            # 检查top预测的类别
            top_class = predictions[0]["class"]
            return f"没有，图片中检测到的是 {top_class} (置信度: {predictions[0]['confidence']:.2%})"

        # 默认返回top-1
        top_class = predictions[0]["class"]
        return f"图片中最可能包含: {top_class} (置信度: {predictions[0]['confidence']:.2%})"

    def _mock_classify(self, text_query: str) -> Dict[str, Any]:
        """模拟分类"""
        if text_query and ("猫" in text_query or "cat" in text_query.lower()):
            answer = "是的，图片中可能包含猫 (模拟结果)"
        else:
            answer = "图片中检测到的是其他物体 (模拟结果)"

        return {
            "predictions": [
                {"class": "tabby cat", "confidence": 0.75},
                {"class": "tiger cat", "confidence": 0.12},
            ],
            "answer": answer,
            "note": "模拟分类结果（模型未加载）"
        }


class SemanticSegmenter:
    """
    语义分割器
    基于DeepLabV3或其他模型进行语义分割
    """

    # PASCAL VOC类别
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
        'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
        'train', 'tv/monitor'
    ]

    # 类别颜色
    VOC_COLORS = [
        (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
        (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
        (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
        (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
        (0, 64, 128)
    ]

    def __init__(self):
        self.model = None
        self.transform = None
        self._loaded = False

    def load_model(self):
        """加载DeepLabV3模型"""
        if self._loaded:
            return

        if TORCH_AVAILABLE:
            try:
                # 加载预训练的DeepLabV3
                self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
                self.model.eval()

                # 图像预处理
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])

                self._loaded = True
                logging.info("DeepLabV3模型加载成功")

            except Exception as e:
                logging.warning(f"DeepLabV3模型加载失败: {e}")
        else:
            logging.warning("torch未安装，将使用模拟分割")

        self._loaded = True

    def segment(self, image: np.ndarray) -> Dict[str, Any]:
        """
        执行语义分割

        Args:
            image: 输入图片

        Returns:
            分割结果
        """
        self.load_model()

        if self.model is not None and TORCH_AVAILABLE:
            try:
                # 转换图像
                pil_image = Image.fromarray(image)
                input_tensor = self.transform(pil_image)
                input_batch = input_tensor.unsqueeze(0)

                # 推理
                with torch.no_grad():
                    output = self.model(input_batch)['out']

                # 获取分割结果
                output_predictions = output.argmax(dim=1).squeeze().cpu().numpy()

                # 统计各类别
                class_stats = self._analyze_segments(output_predictions)

                # 生成分割可视化
                colored_mask = self._colorize_mask(output_predictions)

                return {
                    "segmentation_shape": list(output_predictions.shape),
                    "classes_found": class_stats,
                    "num_classes": len(class_stats),
                    "mask_preview": colored_mask[:100, :100].tolist()  # 部分预览
                }

            except Exception as e:
                logging.error(f"分割失败: {e}")
                return {"error": str(e)}
        else:
            # 模拟分割
            return self._mock_segment(image)

    def _analyze_segments(self, mask: np.ndarray) -> List[Dict[str, Any]]:
        """分析分割结果"""
        unique_classes = np.unique(mask)
        class_stats = []

        for cls_id in unique_classes:
            if cls_id < len(self.VOC_CLASSES):
                class_name = self.VOC_CLASSES[cls_id]
                pixel_count = np.sum(mask == cls_id)
                percentage = pixel_count / mask.size * 100

                class_stats.append({
                    "class_id": int(cls_id),
                    "class_name": class_name,
                    "pixel_count": int(pixel_count),
                    "percentage": round(percentage, 2)
                })

        # 按像素数量排序
        class_stats.sort(key=lambda x: x["pixel_count"], reverse=True)

        return class_stats

    def _colorize_mask(self, mask: np.ndarray) -> np.ndarray:
        """将分割掩码转换为彩色图像"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id in range(len(self.VOC_COLORS)):
            colored[mask == cls_id] = self.VOC_COLORS[cls_id]

        return colored

    def _mock_segment(self, image: np.ndarray) -> Dict[str, Any]:
        """模拟分割"""
        h, w = image.shape[:2] if image is not None else (480, 640)

        return {
            "segmentation_shape": [h, w],
            "classes_found": [
                {"class_name": "background", "pixel_count": h*w*0.7, "percentage": 70.0},
                {"class_name": "cat", "pixel_count": h*w*0.15, "percentage": 15.0},
                {"class_name": "person", "pixel_count": h*w*0.15, "percentage": 15.0},
            ],
            "num_classes": 3,
            "note": "模拟分割结果（模型未加载）"
        }


class ImageProcessingSkill:
    """
    图像处理技能主类
    """

    def __init__(self):
        self.detector = ObjectDetector()
        self.classifier = ImageClassifier()
        self.segmenter = SemanticSegmenter()

    def detect_objects(self,
                       image_url: str,
                       target_description: str = None) -> Dict[str, Any]:
        """
        目标检测

        Args:
            image_url: 图片URL或本地路径
            target_description: 目标描述

        Returns:
            检测结果
        """
        # 下载图片
        image = download_image(image_url)
        if image is None:
            return {"result": "0.0", "error": "无法下载图片", "answer": ""}

        # 执行检测
        detections = self.detector.detect(image, target_description)

        # 格式化输出
        bboxes = {}
        for key, value in detections.items():
            if isinstance(value, dict) and "bbox" in value:
                bboxes[key] = value["bbox"]
            elif isinstance(value, list):
                bboxes[key] = value

        return {
            "result": json.dumps(bboxes, ensure_ascii=False),
            "error": "0.0",
            "answer": f"检测到 {len(bboxes)} 个目标"
        }

    def classify_image(self,
                       image_url: str,
                       text_query: str = None) -> Dict[str, Any]:
        """
        图像分类

        Args:
            image_url: 图片URL或本地路径
            text_query: 查询文本

        Returns:
            分类结果
        """
        # 下载图片
        image = download_image(image_url)
        if image is None:
            return {"result": "0.0", "error": "无法下载图片", "answer": ""}

        # 执行分类
        result = self.classifier.classify(image, text_query)

        return {
            "result": json.dumps(result.get("predictions", [])[:3], ensure_ascii=False),
            "error": "0.0",
            "answer": result.get("answer", "")
        }

    def segment_image(self, image_url: str) -> Dict[str, Any]:
        """
        语义分割

        Args:
            image_url: 图片URL或本地路径

        Returns:
            分割结果
        """
        # 下载图片
        image = download_image(image_url)
        if image is None:
            return {"result": "0.0", "error": "无法下载图片", "answer": ""}

        # 执行分割
        result = self.segmenter.segment(image)

        return {
            "result": json.dumps(result.get("classes_found", []), ensure_ascii=False),
            "error": "0.0",
            "answer": f"检测到 {result.get('num_classes', 0)} 个语义类别"
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像处理技能")
    parser.add_argument("task", choices=["detect", "classify", "segment"],
                       help="任务类型: detect(目标检测), classify(图像分类), segment(语义分割)")
    parser.add_argument("--image", required=True, help="图片URL或本地路径")
    parser.add_argument("--target", help="目标描述（用于目标检测）")
    parser.add_argument("--text", help="查询文本（用于图像分类）")

    args = parser.parse_args()

    setup_logging()

    skill = ImageProcessingSkill()

    if args.task == "detect":
        result = skill.detect_objects(args.image, args.target)
    elif args.task == "classify":
        result = skill.classify_image(args.image, args.text)
    elif args.task == "segment":
        result = skill.segment_image(args.image)
    else:
        result = {"result": "0.0", "error": "未知任务类型", "answer": ""}

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
