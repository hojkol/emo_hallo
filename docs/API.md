# API 文档

Emo Hallo 后端提供 RESTful API 接口用于任务管理和推理。

## 基础信息

- **基础 URL**: `http://localhost:8001`
- **API 前缀**: `/api/v1`
- **内容类型**: `application/json` 或 `multipart/form-data`

## 自动 API 文档

启动后端后，可以访问以下 URL 查看交互式 API 文档：

- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## 端点列表

### 1. 健康检查

#### 简单健康检查
```
GET /health
```

**响应示例：**
```json
{
  "status": "healthy",
  "gpu_available": true,
  "gpu_count": 1,
  "gpu_name": "NVIDIA GeForce RTX 3090"
}
```

#### 详细健康检查
```
GET /api/v1/health
```

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": 3600.5,
  "models_loaded": ["hallo2"],
  "gpu": {
    "available": true,
    "count": 1,
    "devices": [
      {
        "index": 0,
        "name": "NVIDIA GeForce RTX 3090",
        "memory_allocated": "2.5 GB",
        "memory_total": "24 GB"
      }
    ]
  },
  "tasks": {
    "queued": 2,
    "running": 1,
    "completed": 5,
    "failed": 0
  }
}
```

### 2. 推理

#### 创建推理任务
```
POST /api/v1/inference/hallo2
Content-Type: multipart/form-data
```

**请求参数：**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `image_file` | File | ✓ | 源人物图像 (JPG/PNG) |
| `audio_file` | File | ✓ | 驱动音频文件 (WAV/MP3/OGG/M4A) |
| `use_cache` | Boolean | | 是否使用缓存的预处理数据 (默认: true) |

**cURL 示例：**
```bash
curl -X POST "http://localhost:8001/api/v1/inference/hallo2" \
  -F "image_file=@person.jpg" \
  -F "audio_file=@speech.wav" \
  -F "use_cache=true"
```

**Python 示例：**
```python
import requests

files = {
    'image_file': open('person.jpg', 'rb'),
    'audio_file': open('speech.wav', 'rb'),
}
data = {'use_cache': True}

response = requests.post(
    'http://localhost:8001/api/v1/inference/hallo2',
    files=files,
    data=data
)
print(response.json())
```

**成功响应 (200)：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Inference task queued successfully"
}
```

**错误响应：**

- **400 Bad Request**: 文件格式无效
  ```json
  {
    "detail": "Invalid image file format"
  }
  ```

- **500 Internal Server Error**: 提交任务失败
  ```json
  {
    "detail": "Failed to submit task: {error_message}"
  }
  ```

#### 配置推理参数
```
POST /api/v1/inference/hallo2/config
Content-Type: application/json
```

**请求体：**
```json
{
  "guidance_scale": 3.5,
  "inference_steps": 25,
  "enable_cache": true
}
```

**响应示例：**
```json
{
  "status": "success",
  "message": "Configuration updated successfully"
}
```

### 3. 任务管理

#### 获取任务状态
```
GET /api/v1/tasks/{task_id}
```

**URL 参数：**
- `task_id` (string): 任务 ID

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45,
  "status_message": "Running inference...",
  "result": null,
  "error": null,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:00Z"
}
```

**任务状态可能值：**
- `queued` - 任务已排队等待处理
- `processing` - 任务正在处理
- `completed` - 任务已完成
- `failed` - 任务失败

#### 下载生成的视频
```
GET /api/v1/tasks/{task_id}/video
```

**URL 参数：**
- `task_id` (string): 任务 ID

**响应：**
- **200 OK**: MP4 视频文件
- **404 Not Found**: 视频文件不存在
- **400 Bad Request**: 任务未完成

**示例：**
```bash
# 下载视频
curl -O "http://localhost:8001/api/v1/tasks/{task_id}/video"

# 使用 Python
response = requests.get(f'http://localhost:8001/api/v1/tasks/{task_id}/video')
with open('output.mp4', 'wb') as f:
    f.write(response.content)
```

#### 获取任务日志
```
POST /api/v1/tasks/{task_id}/logs
```

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "logs": "[INFO] Starting inference...\n[INFO] Preprocessing image...\n..."
}
```

#### 取消任务
```
DELETE /api/v1/tasks/{task_id}
```

**响应示例：**
```json
{
  "status": "cancelled",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Task cancelled successfully"
}
```

**错误响应：**
```json
{
  "status": "error",
  "detail": "Task not found in queue or already running"
}
```

#### 获取所有任务
```
GET /api/v1/tasks
```

**响应示例：**
```json
{
  "tasks": {
    "550e8400-e29b-41d4-a716-446655440000": {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "progress": 100,
      "status_message": "Task completed",
      ...
    },
    "other-task-id": {...}
  },
  "count": 1
}
```

## 错误处理

所有错误响应遵循标准格式：

```json
{
  "detail": "错误描述信息",
  "error_code": "ERROR_CODE"
}
```

**HTTP 状态码：**

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 400 | 请求错误（参数无效等） |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 使用示例

### 完整工作流

```python
import requests
import time
import json

# 1. 创建推理任务
files = {
    'image_file': open('person.jpg', 'rb'),
    'audio_file': open('speech.wav', 'rb'),
}
response = requests.post(
    'http://localhost:8001/api/v1/inference/hallo2',
    files=files
)
task_data = response.json()
task_id = task_data['task_id']
print(f"Task created: {task_id}")

# 2. 轮询任务状态
while True:
    response = requests.get(f'http://localhost:8001/api/v1/tasks/{task_id}')
    status = response.json()
    print(f"Progress: {status['progress']}% - {status['status_message']}")

    if status['status'] == 'completed':
        print("Task completed!")
        break
    elif status['status'] == 'failed':
        print(f"Task failed: {status['error']}")
        break

    time.sleep(5)  # 每 5 秒检查一次

# 3. 下载视频
if status['status'] == 'completed':
    response = requests.get(
        f'http://localhost:8001/api/v1/tasks/{task_id}/video'
    )
    with open('output.mp4', 'wb') as f:
        f.write(response.content)
    print("Video saved to output.mp4")
```

### 使用客户端库

```python
from emo_hallo.client import Hallo2Client

# 创建客户端
client = Hallo2Client(base_url='http://localhost:8001')

# 创建推理任务
task_id = client.inference(
    image_path='person.jpg',
    audio_path='speech.wav',
    use_cache=True
)

# 等待任务完成并下载
client.wait_and_download(task_id, output_path='output.mp4')
```

## 速率限制

当前版本没有速率限制，但建议：
- 并发任务不超过 `max_concurrent_tasks` 设置
- 单个任务上传文件大小不超过 500MB

## 版本信息

- **API 版本**: v1
- **后端版本**: 1.0.0
- **最后更新**: 2024-01-15

## 更多信息

- 查看 [README.md](../README.md) 了解项目信息
- 查看 [DEVELOPMENT.md](DEVELOPMENT.md) 了解开发指南
- 访问 [GitHub](https://github.com/YOUR_USERNAME/emo_hallo) 获取最新代码
