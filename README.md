# Emo Hallo - Emotional Talking Head Generation

Emo Hallo 是一个基于 Hallo2 模型的情感化虚拟主播生成系统，支持从任意人物照片和音频生成逼真的说话视频。

## 功能特性

- 🎬 **高质量视频生成** - 使用 Hallo2 模型生成自然流畅的说话视频
- 🎤 **音频驱动** - 根据输入音频驱动人物口型和表情
- 💬 **多语言支持** - 中文和英文界面
- 🚀 **异步任务处理** - 支持队列化任务，实时进度更新
- 🌐 **RESTful API** - 提供完整的后端 API 接口
- 💻 **Web UI** - 基于 Streamlit 的直观用户界面

## 项目结构

```
emo_hallo/
├── backend/              # FastAPI 后端服务
│   ├── app.py           # FastAPI 应用入口
│   ├── config.py        # 配置管理
│   ├── controllers/      # API 端点（health, inference, tasks）
│   ├── models/          # 数据模型和常量
│   ├── services/        # 核心服务
│   │   ├── registry.py      # 模型注册表
│   │   ├── task_manager.py  # 任务队列管理
│   │   ├── state_manager.py # 状态管理
│   │   └── hallo2/          # Hallo2 模型和管道
│   └── utils/           # 工具函数（文件、视频、GPU）
├── client/              # HTTP 客户端库
│   └── hallo2_client.py # 后端 API 客户端
├── Main.py             # Streamlit 前端应用
└── i18n/               # 国际化文件
```

## 安装和使用

### 前置要求

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- PyTorch with CUDA support

### 快速开始

```bash
# 克隆仓库
git clone https://github.com/yourusername/emo_hallo.git
cd emo_hallo

# 创建虚拟环境
conda create -n emo_hallo python=3.11
conda activate emo_hallo

# 安装依赖
pip install -r requirements.txt

# 启动服务
bash ../webui2.sh
```

## API 文档

### 健康检查

```
GET /health
GET /api/v1/health
```

### 推理接口

```
POST /api/v1/inference/hallo2
Content-Type: multipart/form-data

Parameters:
  - image_file: 源图像文件 (JPG/PNG)
  - audio_file: 驱动音频文件 (WAV/MP3/OGG/M4A)
  - use_cache: 是否使用缓存 (default: true)

Response:
{
  "task_id": "uuid",
  "status": "queued",
  "message": "Inference task queued successfully"
}
```

### 任务查询

```
GET /api/v1/tasks/{task_id}

Response:
{
  "task_id": "uuid",
  "status": "processing|completed|failed",
  "progress": 0-100,
  "status_message": "...",
  "result": {...},
  "error": null
}
```

### 视频下载

```
GET /api/v1/tasks/{task_id}/video

Returns:
  MP4 video file
```

## 配置

在项目根目录的 `config.toml` 中配置 Hallo2 参数：

```toml
[emo_hallo]
# 模型路径
pose_guide_model = "path/to/pose_guide_model.pth"
image_encoder_model = "path/to/image_encoder.pth"
audio_processor_model = "path/to/audio_processor.pth"
face_locator_model = "path/to/face_locator.pth"

# 推理参数
guidance_scale = 3.5
inference_steps = 25
enable_cache = true
```

## 环境变量

```bash
# 后端环境路径
export EMO_HALLO_BACKEND_ENV=/path/to/conda/env

# 前端环境路径
export EMO_HALLO_FRONTEND_ENV=/path/to/conda/env

# GPU 配置
export EMO_HALLO_GPU_ID=0

# 后端端口
export EMO_HALLO_BACKEND_PORT=8001

# 健康检查配置
export EMO_HALLO_HEALTH_CHECK_RETRIES=20
export EMO_HALLO_HEALTH_CHECK_INTERVAL=60
```

## 文件结构说明

- **logs/** - 系统日志和上传文件（由启动脚本创建）
  - `backend.log` - 后端服务日志
  - `uploads/` - 用户上传的图像、音频及生成的视频
- **.streamlit/** - Streamlit 配置文件
- **i18n/** - 国际化翻译文件

## 开发

### 项目依赖

查看完整依赖列表：
```bash
pip freeze > requirements.txt
```

### 代码风格

遵循 PEP 8 规范。使用 Black 进行代码格式化：

```bash
black emo_hallo/
```

## 常见问题

### 后端无法启动

1. 检查 Python 环境是否正确安装了 PyTorch
2. 查看 `logs/backend.log` 获取详细错误信息
3. 确保 CUDA 驱动和 PyTorch CUDA 版本匹配

### 模型加载缓慢

首次加载模型可能需要 5-10 分钟，这是正常的。之后会使用缓存加速。

### GPU 内存溢出

如果遇到 OOM 错误，尝试：
1. 减小 batch size
2. 使用 float16 精度而非 float32
3. 减少并发任务数

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

- GitHub Issues: 报告 bug 和功能请求
- 项目讨论: 参与技术讨论

## 致谢

基于 Hallo2 模型的出色工作。
