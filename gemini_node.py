"""
ComfyUI Custom Node: Gemini Image Processor
将 Gemini API 集成到 ComfyUI 中用于图片处理

安装方法:
1. 将此文件夹复制到 ComfyUI/custom_nodes/ 目录下
2. 安装依赖: pip install google-generativeai pillow loguru
3. 重启 ComfyUI

Author: Magic Frame Team
Date: 2025-12-17
"""

import os
import time
import asyncio
import numpy as np
import torch
from io import BytesIO
from PIL import Image
from typing import Optional, Tuple, Dict

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("警告: google-generativeai 未安装，Gemini 节点将不可用")
    print("请运行: pip install google-generativeai")


class GeminiImageProcessor:
    """Gemini 图片处理节点"""

    # Gemini 支持的模型列表
    SUPPORTED_MODELS = {
        "gemini-2.5-flash-image-preview": "Gemini 2.5 Flash Image (推荐)",
        "gemini-2.0-flash-exp": "Gemini 2.0 Flash Experimental",
        "gemini-exp-1206": "Gemini Experimental 1206",
    }

    # Gemini 支持的宽高比配置
    SUPPORTED_ASPECT_RATIOS = {
        "auto": "自动检测",
        "1:1": "1:1 (1024x1024)",
        "16:9": "16:9 (1344x768)",
        "9:16": "9:16 (768x1344)",
        "4:3": "4:3 (1184x864)",
        "3:4": "3:4 (864x1184)",
        "21:9": "21:9 (1536x672)",
    }

    # 预设提示词
    PRESET_PROMPTS = {
        "black": "黑白照片上色",
        "old": "老照片修复",
        "real": "现代照片增强",
        "others": "特殊照片处理",
        "custom": "自定义提示词",
    }

    def __init__(self):
        self.client = None
        self.api_key = None

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入参数"""
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图片
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "输入 Gemini API Key 或留空使用环境变量"
                }),
                "model": (list(cls.SUPPORTED_MODELS.keys()),),  # 模型选择
                "prompt_preset": (list(cls.PRESET_PROMPTS.keys()),),  # 预设提示词
                "aspect_ratio": (list(cls.SUPPORTED_ASPECT_RATIOS.keys()),),  # 宽高比
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
            },
            "optional": {
                "custom_prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "当选择'自定义提示词'时使用"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "process_image"
    CATEGORY = "Gemini"

    def tensor_to_pil(self, tensor):
        """将 ComfyUI 的 tensor 转换为 PIL Image"""
        # tensor shape: [B, H, W, C]
        # 取第一张图片
        img = tensor[0]
        # 转换为 numpy array
        img = (img.cpu().numpy() * 255).astype(np.uint8)
        # 转换为 PIL Image
        return Image.fromarray(img)

    def pil_to_tensor(self, pil_image):
        """将 PIL Image 转换为 ComfyUI 的 tensor"""
        # 转换为 numpy array
        img = np.array(pil_image).astype(np.float32) / 255.0
        # 添加 batch 维度
        img = torch.from_numpy(img)[None,]
        return img

    def get_prompt_by_preset(self, preset: str, custom_prompt: str = "") -> str:
        """根据预设获取提示词"""
        if preset == "custom" and custom_prompt:
            return custom_prompt

        prompts = {
            "black": """*** 系统指令：黑白照片智能上色与修复 ***
你是一位专业的黑白照片上色和修复专家。你的任务是将黑白/灰白照片转换为自然真实的彩色照片。

### 🎯 核心任务：
1. **智能上色**（最重要）：根据照片年代还原符合时代特征的色彩，人物肤色自然真实
2. **损伤修复**：去除划痕、污点、折痕、霉斑，修复破损区域
3. **细节增强**：提升清晰度，增强面部细节

*** 输出要求：直接输出修复并上色后的高清彩色图像 ***""",

            "old": """*** 系统指令：老照片修复与色彩还原 ***
你是一位专业的老照片修复专家。这是一张有色彩但已经严重老化的照片。

### 🎯 核心任务：
1. **重度损伤修复**：去除严重的划痕、裂纹、破损、霉斑、水渍
2. **色彩修复**（最重要）：校正严重的泛黄、泛红、褪色问题，还原真实色彩
3. **画质提升**：大幅提升清晰度和分辨率

*** 输出要求：直接输出全面修复并色彩还原后的高清图像 ***""",

            "real": """*** 系统指令：现代照片质量增强 ***
你是一位专业的照片质量优化专家。这是一张质量较好的现代照片。

### 🎯 核心任务：
1. **画质优化**：轻微提升清晰度，优化对比度和亮度
2. **细微瑕疵修复**：去除小污点、灰尘，修正轻微色偏
3. **专业润色**：使照片达到专业摄影水准

⚠️ 克制为上：不要过度处理，保持照片真实感

*** 输出要求：直接输出适度优化后的高清图像 ***""",

            "others": """*** 系统指令：特殊照片智能处理 ***
你是一位专业的照片修复和优化专家。这是一张特殊类型的照片。

### 🎯 智能分析与处理：
1. **智能识别照片类型**：自动识别相框、证件照、艺术照等特殊类型
2. **针对性修复**：根据照片类型选择合适的修复强度
3. **风格保持**：保留原照片的特殊风格和特征

*** 输出要求：直接输出智能修复并优化后的高清图像 ***""",
        }

        return prompts.get(preset, prompts["old"])

    def init_client(self, api_key: str):
        """初始化 Gemini 客户端"""
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai 未安装，请运行: pip install google-generativeai")

        # 获取 API Key
        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("未配置 GEMINI_API_KEY，请在节点中输入或设置环境变量")

        # 如果 API Key 变化，重新初始化
        if self.api_key != api_key:
            self.api_key = api_key
            self.client = genai.Client(api_key=self.api_key)
            print("[Gemini] 客户端初始化成功")

    def calculate_aspect_ratio(self, width: int, height: int) -> str:
        """计算最佳宽高比"""
        if height == 0:
            return "1:1"
        ratio = width / height

        # 找到最接近的宽高比
        ratio_map = {
            "1:1": 1.0,
            "16:9": 1.75,
            "9:16": 0.5714,
            "4:3": 1.3704,
            "3:4": 0.7297,
            "21:9": 2.2857,
        }

        best_ratio = "1:1"
        min_diff = float('inf')
        for ratio_name, ratio_value in ratio_map.items():
            diff = abs(ratio_value - ratio)
            if diff < min_diff:
                min_diff = diff
                best_ratio = ratio_name

        return best_ratio

    async def generate_image_async(
        self,
        image_data: bytes,
        prompt: str,
        model: str,
        aspect_ratio: str,
        temperature: float
    ) -> Tuple[bytes, str]:
        """异步生成图片，返回图片数据和文本输出"""
        # 将图片数据转换为 PIL Image
        image = Image.open(BytesIO(image_data))

        # 如果是自动检测，计算最佳宽高比
        if aspect_ratio == "auto":
            width, height = image.size
            aspect_ratio = self.calculate_aspect_ratio(width, height)
            print(f"[Gemini] 自动检测宽高比: {aspect_ratio}")

        # 安全设置
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ]

        # 调用 Gemini API
        print(f"[Gemini] 使用模型: {model}")
        response = self.client.models.generate_content(
            model=model,
            contents=[prompt, image],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
                safety_settings=safety_settings,
                temperature=temperature,
                top_p=1.0,
                top_k=1,
            )
        )

        # 检查响应
        if not response.candidates or len(response.candidates) == 0:
            raise ValueError("Gemini API 未返回有效结果")

        candidate = response.candidates[0]

        # 获取 finish_reason
        finish_reason = "未知"
        if hasattr(candidate, 'finish_reason'):
            finish_reason = str(candidate.finish_reason)
            print(f"[Gemini] finish_reason: {finish_reason}")

        if not hasattr(candidate, 'content') or candidate.content is None:
            raise ValueError(f"Gemini 返回的内容为空，finish_reason: {finish_reason}")

        # 提取图片数据和文本
        result_image_data = None
        text_output = ""

        for part in candidate.content.parts:
            # 提取文本
            if part.text is not None:
                text_output += part.text
                print(f"[Gemini] AI 输出文本: {part.text}")

            # 提取图片数据
            if part.inline_data is not None:
                result_image_data = part.inline_data.data

        # 如果没有图片数据
        if not result_image_data:
            error_msg = f"AI 模型未返回有效图片数据"
            if text_output:
                error_msg += f"\n模型返回文本: {text_output}"
            error_msg += f"\nfinish_reason: {finish_reason}"
            raise ValueError(error_msg)

        # 构建完整的文本输出（包含元信息）
        full_text = f"模型: {model}\n"
        full_text += f"宽高比: {aspect_ratio}\n"
        full_text += f"finish_reason: {finish_reason}\n"
        if text_output:
            full_text += f"\nAI 输出:\n{text_output}"
        else:
            full_text += f"\nAI 输出: (仅返回图片，无文本输出)"

        return result_image_data, full_text

    def process_image(
        self,
        image,
        api_key: str,
        model: str,
        prompt_preset: str,
        aspect_ratio: str,
        temperature: float,
        custom_prompt: str = ""
    ):
        """处理图片（主函数）"""
        try:
            print(f"[Gemini] 开始处理图片...")
            print(f"[Gemini] 模型: {model}, 预设: {prompt_preset}, 宽高比: {aspect_ratio}, 温度: {temperature}")

            # 初始化客户端
            self.init_client(api_key)

            # 将 tensor 转换为 PIL Image
            pil_image = self.tensor_to_pil(image)

            # 转换为字节流
            img_byte_arr = BytesIO()
            pil_image.save(img_byte_arr, format='PNG')
            image_data = img_byte_arr.getvalue()

            # 获取提示词
            prompt = self.get_prompt_by_preset(prompt_preset, custom_prompt)

            # 异步调用 Gemini API
            start_time = time.time()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result_data, text_output = loop.run_until_complete(
                self.generate_image_async(image_data, prompt, model, aspect_ratio, temperature)
            )
            loop.close()

            elapsed = time.time() - start_time
            print(f"[Gemini] 处理完成，耗时: {elapsed:.2f}秒")

            # 将结果转换回 PIL Image
            result_image = Image.open(BytesIO(result_data))

            # 转换为 ComfyUI tensor
            result_tensor = self.pil_to_tensor(result_image)

            # 返回图片和文本
            return (result_tensor, text_output)

        except Exception as e:
            error_msg = f"Gemini 处理失败: {str(e)}"
            print(f"[Gemini] 错误: {error_msg}")
            raise RuntimeError(error_msg)


# ComfyUI 节点注册
NODE_CLASS_MAPPINGS = {
    "GeminiImageProcessor": GeminiImageProcessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiImageProcessor": "Gemini Clond api Image Processor",
}

