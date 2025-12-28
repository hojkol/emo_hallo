# 开发指南

本文档包含 Emo Hallo 的开发人员指南，包括如何设置开发环境、运行测试、构建和部署。

## 目录

- [开发环境设置](#开发环境设置)
- [项目结构](#项目结构)
- [运行后端](#运行后端)
- [运行前端](#运行前端)
- [测试](#测试)
- [调试](#调试)
- [构建与发布](#构建与发布)

## 开发环境设置

### 前置条件

- Python 3.9+
- Conda（推荐）或 pip/venv
- CUDA 11.0+（用于 GPU 支持）
- Git

### 步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/YOUR_USERNAME/emo_hallo.git
   cd emo_hallo
   ```

2. **创建 Conda 环境**
   ```bash
   conda create -n emo_hallo-dev python=3.11
   conda activate emo_hallo-dev
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 mypy
   ```

4. **验证安装**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
   python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
   ```

## 项目结构

```
emo_hallo/
├── backend/
│   ├── app.py                    # FastAPI 应用入口
│   ├── config/
│   │   └── settings.py          # 配置管理
│   ├── controllers/              # API 端点
│   ├── models/                   # 数据模型
│   ├── services/                 # 核心业务逻辑
│   │   ├── registry/            # 模型注册表
│   │   ├── hallo2/              # Hallo2 模型和管道
│   │   └── base/                # 基类接口
│   └── utils/                    # 工具函数
├── client/
│   └── hallo2_client.py         # HTTP 客户端
├── tests/                        # 测试用例
├── Main.py                       # Streamlit 前端
├── requirements.txt              # 依赖列表
└── README.md                     # 项目说明
```

## 运行后端

### 开发模式（带热加载）

```bash
cd /path/to/moneyprinterturbo
python -m uvicorn emo_hallo.backend.app:app --reload --host 0.0.0.0 --port 8001
```

### 生产模式（使用 webui2.sh）

```bash
bash webui2.sh
```

### 验证后端

```bash
# 健康检查
curl http://localhost:8001/health

# API 文档
open http://localhost:8001/docs  # 自动生成的 Swagger UI
```

## 运行前端

### 开发模式

```bash
streamlit run emo_hallo/Main.py
```

访问 `http://localhost:8501`

### 环境变量

```bash
export BACKEND_URL=http://localhost:8001
streamlit run emo_hallo/Main.py
```

## 测试

### 运行所有测试

```bash
pytest tests/ -v
```

### 运行特定测试

```bash
# 运行特定文件
pytest tests/test_backend.py -v

# 运行特定测试类
pytest tests/test_backend.py::TestInference -v

# 运行特定测试
pytest tests/test_backend.py::TestInference::test_inference -v
```

### 生成覆盖率报告

```bash
pytest tests/ --cov=backend --cov=client --cov-report=html
open htmlcov/index.html
```

### 异步测试

```bash
pytest tests/ -v --asyncio-mode=auto
```

## 调试

### 后端调试

**使用 print 调试**
```python
import logging

logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {variable}")
```

**查看日志**
```bash
# 实时查看日志
tail -f logs/backend.log

# 搜索特定错误
grep "ERROR" logs/backend.log
```

**使用 PyCharm/VSCode 调试器**

1. 在 `backend/app.py` 中设置断点
2. 运行调试器
3. 访问 `http://localhost:8001/health` 触发断点

**使用 pdb 调试**
```python
import pdb; pdb.set_trace()  # 在代码中设置断点
```

### 前端调试

**Streamlit 调试模式**
```bash
streamlit run emo_hallo/Main.py --logger.level=debug
```

**查看组件状态**
```python
import streamlit as st
st.write(st.session_state)
```

## 代码质量

### 代码格式化

```bash
# 使用 Black 格式化
black emo_hallo/

# 查看格式化前的差异
black --diff emo_hallo/
```

### 代码检查

```bash
# Flake8 检查
flake8 emo_hallo/ --max-line-length=127

# Pylint 检查
pylint emo_hallo/

# 类型检查（可选）
mypy emo_hallo/ --ignore-missing-imports
```

### 预提交检查（推荐）

```bash
# 安装 pre-commit
pip install pre-commit

# 创建 .pre-commit-config.yaml
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length=127']
EOF

# 安装 git hooks
pre-commit install

# 手动运行（可选）
pre-commit run --all-files
```

## 构建与发布

### 创建发布版本

```bash
# 更新版本号
vim emo_hallo/__init__.py  # 修改 __version__

# 创建 Git tag
git tag v1.0.0
git push origin v1.0.0
```

### 发布到 PyPI（可选）

```bash
# 安装 build 工具
pip install build twine

# 构建分发包
python -m build

# 上传到 PyPI
python -m twine upload dist/*
```

## 常见问题

### Q: 如何重置开发环境？
```bash
# 删除环境
conda env remove --name emo_hallo-dev

# 重新创建
conda create -n emo_hallo-dev python=3.11
conda activate emo_hallo-dev
pip install -r requirements.txt
```

### Q: 如何清理 Python 缓存？
```bash
# 删除 __pycache__ 目录
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 删除 .pyc 文件
find . -type f -name "*.pyc" -delete
```

### Q: 如何查看 GPU 使用情况？
```bash
# NVIDIA GPU
nvidia-smi

# 监控 PyTorch GPU 使用
python -c "import torch; print(torch.cuda.memory_allocated() / 1e9, 'GB')"
```

### Q: 如何远程调试？
```bash
# 在服务器上运行（带有调试服务）
python -m pdb backend/app.py

# 或使用远程 IDE 配置
```

## 资源

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Streamlit 官方文档](https://docs.streamlit.io/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [Python 测试最佳实践](https://docs.python-guide.org/writing/tests/)

## 获取帮助

- 查看 [README.md](../README.md)
- 查看 [CONTRIBUTING.md](../CONTRIBUTING.md)
- 创建 [GitHub Issue](https://github.com/YOUR_USERNAME/emo_hallo/issues)
