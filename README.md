# ComfyUI Gemini 图片处理节点

将 Gemini API 集成到 ComfyUI 中，用于老照片修复、黑白照片上色、图片增强等任务。

## 功能特性

- ✅ **无缝集成**：作为 ComfyUI 自定义节点使用
- ✅ **预设提示词**：内置4种专用提示词（黑白上色、老照片修复、现代增强、特殊处理）
- ✅ **自定义提示词**：支持自定义提示词
- ✅ **宽高比控制**：支持自动检测或手动指定输出宽高比
- ✅ **温度控制**：可调节输出的随机性

## 安装步骤

### 1. 复制文件到 ComfyUI

将 `comfyui_nodes` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录：

```bash
# 方法1：直接复制
cp -r comfyui_nodes /path/to/ComfyUI/custom_nodes/gemini_processor

# 方法2：创建符号链接（开发时推荐）
ln -s /path/to/Photo_colorizer/ph/nano/comfyui_nodes /path/to/ComfyUI/custom_nodes/gemini_processor
```


### 2. 安装依赖

在 ComfyUI 的 Python 环境中安装依赖：

```bash
# 进入 ComfyUI 目录
cd /path/to/ComfyUI

# 安装依赖
pip install google-genai pillow loguru
```

### 3. 设置 API Key

有两种方式设置 Gemini API Key：

**方法1：环境变量（推荐）**
```bash
# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"

# Windows PowerShell
$env:GEMINI_API_KEY="your_api_key_here"

# Windows CMD
set GEMINI_API_KEY=your_api_key_here
```

**方法2：在节点中输入**
- 在 ComfyUI 界面中，直接在节点的 `api_key` 参数中输入

### 4. 重启 ComfyUI

重启 ComfyUI 以加载新节点。

## 使用方法

### 在 ComfyUI 中添加节点

1. 在 ComfyUI 中，右键点击空白处
2. 选择 `Add Node` -> `Gemini` -> `Gemini 图片处理器`
3. 或者搜索 "Gemini"

### 节点参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| **image** | IMAGE | 输入图片（必需） |
| **api_key** | STRING | Gemini API Key，留空则使用环境变量 |
| **model** | ENUM | 模型选择（3个可选模型） |
| **prompt_preset** | ENUM | 预设提示词类型 |
| **aspect_ratio** | ENUM | 输出宽高比 |
| **temperature** | FLOAT | 温度参数（0-1），控制输出随机性 |
| **custom_prompt** | STRING | 自定义提示词（可选） |

### 节点输出说明

| 输出 | 类型 | 说明 |
|------|------|------|
| **image** | IMAGE | 处理后的图片 |
| **text** | STRING | AI 输出的文本信息（包含模型、宽高比、finish_reason 等） |

### 预设提示词类型

| 类型 | 说明 | 适用场景 |
|------|------|----------|
| **black** | 黑白照片上色 | 黑白或灰白色调的老照片 |
| **old** | 老照片修复 | 有色彩但严重褪色、损坏的照片 |
| **real** | 现代照片增强 | 质量较好的现代照片 |
| **others** | 特殊照片处理 | 带相框、证件照等特殊类型 |
| **custom** | 自定义提示词 | 使用自定义的提示词 |

### 模型选择

| 模型 | 说明 | 推荐场景 |
|------|------|----------|
| **gemini-2.5-flash-image-preview** | Gemini 2.5 Flash Image（推荐） | 图片处理、老照片修复、上色等任务 |
| **gemini-2.0-flash-exp** | Gemini 2.0 Flash Experimental | 实验性功能，可能有新特性 |

### 宽高比选项

| 选项 | 分辨率 | 说明 |
|------|--------|------|
| **auto** | 自动 | 根据输入图片自动选择最佳比例 |
| **1:1** | 1024×1024 | 正方形 |
| **16:9** | 1344×768 | 横向宽屏 |
| **9:16** | 768×1344 | 竖向手机屏 |
| **4:3** | 1184×864 | 标准横向 |
| **3:4** | 864×1184 | 标准竖向 |
| **21:9** | 1536×672 | 超宽屏 |

## 工作流示例

### 示例1：基本老照片修复（带文本输出）

```
Load Image -> Gemini 图片处理器 -> Save Image
              ↓                    ↓
              参数:                text -> Show Text 或 Save Text
              - model: gemini-2.5-flash-image-preview
              - preset: old
              - aspect_ratio: auto
              - temperature: 0
```

### 示例2：黑白照片上色

```
Load Image -> Gemini 图片处理器 -> Save Image
              ↓
              参数:
              - model: gemini-2.5-flash-image-preview
              - preset: black
              - aspect_ratio: auto
              - temperature: 0
```

### 示例3：批量处理

```
Load Images (Batch) -> Gemini 图片处理器 -> Save Images
                       ↓
                       参数:
                       - model: gemini-2.5-flash-image-preview
                       - preset: old
                       - aspect_ratio: auto
                       - temperature: 0
```

### 示例4：自定义提示词 + 文本输出

```
Load Image -> Gemini 图片处理器 -> Save Image
              ↓                    ↓
              参数:                text -> Show Text
              - model: gemini-exp-1206
              - preset: custom
              - custom_prompt: "转换为油画风格"
              - aspect_ratio: 16:9
              - temperature: 0.5
```

## 工作流 JSON 示例

```json
{
  "1": {
    "inputs": {
      "image": "example.jpg",
      "upload": "image"
    },
    "class_type": "LoadImage"
  },
  "2": {
    "inputs": {
      "image": ["1", 0],
      "api_key": "",
      "model": "gemini-2.5-flash-image-preview",
      "prompt_preset": "old",
      "aspect_ratio": "auto",
      "temperature": 0.0,
      "custom_prompt": ""
    },
    "class_type": "GeminiImageProcessor"
  },
  "3": {
    "inputs": {
      "images": ["2", 0],
      "filename_prefix": "gemini_output"
    },
    "class_type": "SaveImage"
  },
  "4": {
    "inputs": {
      "text": ["2", 1]
    },
    "class_type": "ShowText"
  }
}
```

## 常见问题

### Q: 节点没有显示？
A: 
1. 确认文件夹复制到了正确位置
2. 确认 `__init__.py` 文件存在
3. 重启 ComfyUI
4. 查看 ComfyUI 控制台是否有错误信息

### Q: 提示 "google-generativeai 未安装"？
A: 在 ComfyUI 的 Python 环境中运行：
```bash
pip install google-generativeai pillow loguru
```

### Q: 提示 "未配置 GEMINI_API_KEY"？
A: 
1. 设置环境变量，或
2. 在节点的 `api_key` 参数中直接输入

### Q: 处理速度慢？
A: 
1. Gemini API 调用需要网络请求，通常需要 5-10 秒
2. 确保网络连接稳定
3. 如果在国内，可能需要使用代理

### Q: 如何使用代理？
A: 在启动 ComfyUI 前设置代理环境变量：
```bash
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### Q: 支持批量处理吗？
A: 支持！ComfyUI 会自动处理批量输入的每张图片。

## 技术细节

### 节点类型
- **分类**：Gemini
- **输入类型**：IMAGE（ComfyUI 标准格式）
- **输出类型**：IMAGE（ComfyUI 标准格式）
- **执行函数**：process_image

### 图片格式转换
- ComfyUI Tensor (BHWC) -> PIL Image -> Gemini API
- Gemini API -> PIL Image -> ComfyUI Tensor (BHWC)

### 错误处理
- 所有错误都会在 ComfyUI 控制台显示
- API 调用失败会抛出 RuntimeError
- 节点会保留输入图片以便调试

## 更新日志

### v1.0.0 (2025-12-17)
- ✅ 初始版本
- ✅ 支持4种预设提示词
- ✅ 支持自定义提示词
- ✅ 支持宽高比控制
- ✅ 支持温度参数调节

## License

MIT License

## 相关链接

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [Google Gemini API](https://ai.google.dev/)
- [自定义节点开发文档](https://github.com/comfyanonymous/ComfyUI/wiki/Custom-Nodes)

