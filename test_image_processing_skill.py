#!/usr/bin/env python3
"""
图像处理 Skill 单元测试
测试目标检测、图像分类、语义分割三个任务

运行方式：
cd /Users/guyanlei/Desktop/图像处理
python test_image_processing_skill.py
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import numpy as np

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入被测试模块
from image_processing_skill import (
    ObjectDetector,
    ImageClassifier,
    SemanticSegmenter,
    ImageProcessingSkill,
    download_image
)


class TestDownloadImage(unittest.TestCase):
    """测试图片下载功能"""

    def test_download_image_invalid_url(self):
        """测试无效URL"""
        result = download_image("invalid_url")
        self.assertIsNone(result)

    def test_download_image_local_file(self):
        """测试本地文件"""
        # 创建临时测试图片
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # 创建一个简单的测试图片
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            temp_path = f.name

        try:
            result = download_image(temp_path)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, (100, 100, 3))
        finally:
            os.unlink(temp_path)


class TestObjectDetector(unittest.TestCase):
    """测试目标检测器"""

    def setUp(self):
        """测试前准备"""
        self.detector = ObjectDetector()
        # 创建测试图片
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector._loaded)

    def test_load_model(self):
        """测试模型加载"""
        self.detector.load_model()
        self.assertTrue(self.detector._loaded)

    def test_detect_without_target(self):
        """测试无目标描述的检测"""
        self.detector.load_model()
        results = self.detector.detect(self.test_image)
        self.assertIsInstance(results, dict)

    def test_detect_with_cat_target(self):
        """测试指定猫的检测"""
        self.detector.load_model()
        results = self.detector.detect(self.test_image, target_description="猫")
        self.assertIsInstance(results, dict)

    def test_detect_with_dog_target(self):
        """测试指定狗的检测"""
        self.detector.load_model()
        results = self.detector.detect(self.test_image, target_description="狗")
        self.assertIsInstance(results, dict)

    def test_parse_target_description(self):
        """测试目标描述解析"""
        # 测试中文
        targets = self.detector._parse_target_description("检测图片中的猫")
        self.assertIn("cat", targets)

        targets = self.detector._parse_target_description("找到狗")
        self.assertIn("dog", targets)

        # 测试英文
        targets = self.detector._parse_target_description("find the car")
        self.assertIn("car", targets)

    def test_mock_detect(self):
        """测试模拟检测"""
        results = self.detector._mock_detect(self.test_image, "猫")
        self.assertIn("0", results)
        self.assertIn("bbox", results["0"])


class TestImageClassifier(unittest.TestCase):
    """测试图像分类器"""

    def setUp(self):
        """测试前准备"""
        self.classifier = ImageClassifier()
        # 创建测试图片
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.classifier)
        self.assertFalse(self.classifier._loaded)

    def test_load_model(self):
        """测试模型加载"""
        self.classifier.load_model()
        self.assertTrue(self.classifier._loaded)

    def test_classify_without_query(self):
        """测试无查询文本的分类"""
        self.classifier.load_model()
        result = self.classifier.classify(self.test_image)
        self.assertIn("predictions", result)
        self.assertIn("answer", result)

    def test_classify_with_cat_query(self):
        """测试查询猫的分类"""
        self.classifier.load_model()
        result = self.classifier.classify(self.test_image, text_query="是否有猫")
        self.assertIn("answer", result)

    def test_classify_with_dog_query(self):
        """测试查询狗的分类"""
        self.classifier.load_model()
        result = self.classifier.classify(self.test_image, text_query="是否有狗")
        self.assertIn("answer", result)

    def test_mock_classify(self):
        """测试模拟分类"""
        result = self.classifier._mock_classify("是否有猫")
        self.assertIn("predictions", result)
        self.assertIn("answer", result)


class TestSemanticSegmenter(unittest.TestCase):
    """测试语义分割器"""

    def setUp(self):
        """测试前准备"""
        self.segmenter = SemanticSegmenter()
        # 创建测试图片
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.segmenter)
        self.assertFalse(self.segmenter._loaded)

    def test_load_model(self):
        """测试模型加载"""
        self.segmenter.load_model()
        self.assertTrue(self.segmenter._loaded)

    def test_segment(self):
        """测试语义分割"""
        self.segmenter.load_model()
        result = self.segmenter.segment(self.test_image)
        self.assertIn("classes_found", result)

    def test_analyze_segments(self):
        """测试分割结果分析"""
        # 创建模拟分割掩码
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[:50, :] = 0  # background
        mask[50:, :50] = 8  # cat
        mask[50:, 50:] = 15  # person

        class_stats = self.segmenter._analyze_segments(mask)
        self.assertEqual(len(class_stats), 3)

    def test_colorize_mask(self):
        """测试掩码着色"""
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[:50, :] = 0
        mask[50:, :] = 8

        colored = self.segmenter._colorize_mask(mask)
        self.assertEqual(colored.shape, (100, 100, 3))

    def test_mock_segment(self):
        """测试模拟分割"""
        result = self.segmenter._mock_segment(self.test_image)
        self.assertIn("classes_found", result)
        self.assertIn("num_classes", result)


class TestImageProcessingSkill(unittest.TestCase):
    """测试图像处理技能主类"""

    def setUp(self):
        """测试前准备"""
        self.skill = ImageProcessingSkill()

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.skill.detector)
        self.assertIsNotNone(self.skill.classifier)
        self.assertIsNotNone(self.skill.segmenter)

    def test_detect_objects_invalid_url(self):
        """测试无效URL的目标检测"""
        result = self.skill.detect_objects("invalid_url", "猫")
        self.assertEqual(result["error"], "无法下载图片")

    def test_classify_image_invalid_url(self):
        """测试无效URL的图像分类"""
        result = self.skill.classify_image("invalid_url", "是否有猫")
        self.assertEqual(result["error"], "无法下载图片")

    def test_segment_image_invalid_url(self):
        """测试无效URL的语义分割"""
        result = self.skill.segment_image("invalid_url")
        self.assertEqual(result["error"], "无法下载图片")

    def test_output_format(self):
        """测试输出格式"""
        # 创建临时测试图片
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            from PIL import Image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            temp_path = f.name

        try:
            # 测试检测输出格式
            result = self.skill.detect_objects(temp_path, "猫")
            self.assertIn("result", result)
            self.assertIn("error", result)
            self.assertIn("answer", result)

            # 测试分类输出格式
            result = self.skill.classify_image(temp_path, "是否有猫")
            self.assertIn("result", result)
            self.assertIn("error", result)
            self.assertIn("answer", result)

            # 测试分割输出格式
            result = self.skill.segment_image(temp_path)
            self.assertIn("result", result)
            self.assertIn("error", result)
            self.assertIn("answer", result)

        finally:
            os.unlink(temp_path)


class TestWithMockImage(unittest.TestCase):
    """使用模拟图片的集成测试"""

    @classmethod
    def setUpClass(cls):
        """创建测试图片"""
        cls.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 保存临时文件
        cls.temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        from PIL import Image
        img = Image.fromarray(cls.test_image)
        img.save(cls.temp_file.name)
        cls.temp_path = cls.temp_file.name

    @classmethod
    def tearDownClass(cls):
        """清理临时文件"""
        os.unlink(cls.temp_path)

    def test_full_detection_pipeline(self):
        """完整检测流水线测试"""
        skill = ImageProcessingSkill()
        result = skill.detect_objects(self.temp_path, "猫")

        print("\n=== 目标检测结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        self.assertIn("result", result)
        self.assertIn("answer", result)

    def test_full_classification_pipeline(self):
        """完整分类流水线测试"""
        skill = ImageProcessingSkill()
        result = skill.classify_image(self.temp_path, "是否有猫")

        print("\n=== 图像分类结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        self.assertIn("result", result)
        self.assertIn("answer", result)

    def test_full_segmentation_pipeline(self):
        """完整分割流水线测试"""
        skill = ImageProcessingSkill()
        result = skill.segment_image(self.temp_path)

        print("\n=== 语义分割结果 ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

        self.assertIn("result", result)
        self.assertIn("answer", result)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDownloadImage))
    suite.addTests(loader.loadTestsFromTestCase(TestObjectDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestImageClassifier))
    suite.addTests(loader.loadTestsFromTestCase(TestSemanticSegmenter))
    suite.addTests(loader.loadTestsFromTestCase(TestImageProcessingSkill))
    suite.addTests(loader.loadTestsFromTestCase(TestWithMockImage))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


def run_demo():
    """运行演示测试"""
    print("=" * 60)
    print("图像处理 Skill 演示测试")
    print("=" * 60)

    # 使用测试图片URL
    test_url = "https://pics7.baidu.com/feed/09fa513d269759ee183c20de87d927196d22df6c.jpeg@f_auto?token=47985a0a0ad265709ff4ee6d44ed7b9b"

    skill = ImageProcessingSkill()

    # 1. 目标检测
    print("\n【任务1：目标检测】")
    print(f"输入: 图片URL, 目标='猫'")
    result = skill.detect_objects(test_url, "猫")
    print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # 2. 图像分类
    print("\n【任务2：图像分类】")
    print(f"输入: 图片URL, 查询='是否有猫'")
    result = skill.classify_image(test_url, "是否有猫")
    print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")

    # 3. 语义分割
    print("\n【任务3：语义分割】")
    print(f"输入: 图片URL")
    result = skill.segment_image(test_url)
    print(f"输出: {json.dumps(result, ensure_ascii=False, indent=2)}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="图像处理 Skill 测试")
    parser.add_argument("--demo", action="store_true", help="运行演示测试")
    parser.add_argument("--unit", action="store_true", help="运行单元测试")

    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.unit:
        run_tests()
    else:
        # 默认运行演示
        print("运行演示测试 (使用 --unit 运行单元测试)\n")
        run_demo()
