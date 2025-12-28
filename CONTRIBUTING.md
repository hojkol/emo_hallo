# 贡献指南

感谢你对 Emo Hallo 项目的兴趣！我们欢迎各种形式的贡献，包括 Bug 报告、功能请求、文档改进和代码提交。

## 如何贡献

### 报告 Bug

如果你发现了 Bug，请创建一个 Issue 并包含以下信息：

- **问题描述** - 清晰描述你遇到的问题
- **复现步骤** - 如何重现这个问题的详细步骤
- **实际行为** - 实际发生了什么
- **预期行为** - 应该发生什么
- **环境信息**：
  - Python 版本
  - PyTorch 版本
  - CUDA 版本
  - 操作系统
  - GPU 型号
- **错误日志** - 完整的错误堆栈跟踪（在 `logs/backend.log` 中）
- **屏幕截图** - 如果适用的话

### 请求新功能

创建 Issue 时请包括：

- **功能描述** - 新功能是什么
- **使用场景** - 为什么需要这个功能
- **建议实现** - 你有什么想法吗？（可选）
- **相关 Issue** - 是否与其他 Issue 相关

### 提交代码

1. **Fork 项目**
   ```bash
   git clone https://github.com/YOUR_USERNAME/emo_hallo.git
   cd emo_hallo
   ```

2. **创建特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   # 或修复 Bug
   git checkout -b fix/bug-description
   ```

3. **安装开发依赖**
   ```bash
   conda create -n emo_hallo-dev python=3.11
   conda activate emo_hallo-dev
   pip install -r requirements.txt
   pip install black flake8 pytest
   ```

4. **进行更改**
   - 修改代码
   - 添加/更新测试
   - 更新文档

5. **验证代码质量**
   ```bash
   # 代码格式化
   black emo_hallo/

   # 代码风格检查
   flake8 emo_hallo/ --max-line-length=100

   # 运行测试（如果有）
   pytest tests/
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "feat: 添加新功能描述"
   # 或
   git commit -m "fix: 修复 Bug 描述"
   ```

7. **推送到你的 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 清晰描述你的更改
   - 链接相关的 Issue

## 代码规范

### Python 代码风格

遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 规范：

```python
# 良好示例
def process_image(image_path: str, config: Dict[str, Any]) -> np.ndarray:
    """
    处理输入图像。

    Args:
        image_path: 图像文件路径
        config: 配置字典

    Returns:
        处理后的图像数组

    Raises:
        FileNotFoundError: 如果图像文件不存在
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 处理逻辑
    return processed_image
```

### 命名规范

- **类名** - PascalCase：`class Hallo2Model`
- **函数名** - snake_case：`def process_image()`
- **常量** - UPPER_SNAKE_CASE：`MAX_RETRIES = 20`
- **私有成员** - 前缀 `_`：`_internal_cache`

### 文档字符串

使用 Google 风格的文档字符串：

```python
def inference(
    image_path: str,
    audio_path: str,
    save_dir: str,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    运行 Hallo2 推理。

    Args:
        image_path: 源图像路径
        audio_path: 驱动音频路径
        save_dir: 结果保存目录
        progress_callback: 进度回调函数

    Returns:
        包含以下键的字典：
            - video_path (str): 生成的视频路径
            - duration (float): 视频时长（秒）
            - frames (int): 视频帧数

    Raises:
        ValueError: 如果输入文件格式不正确
        RuntimeError: 如果推理过程失败

    Example:
        >>> result = inference(
        ...     "image.jpg",
        ...     "audio.wav",
        ...     "output_dir"
        ... )
        >>> print(result['video_path'])
    """
```

### 错误处理

```python
# 好的做法
try:
    model = load_model(model_path)
except FileNotFoundError as e:
    logger.error(f"Model file not found: {model_path}")
    raise RuntimeError(f"Failed to load model: {str(e)}") from e
except Exception as e:
    logger.error(f"Unexpected error loading model: {str(e)}", exc_info=True)
    raise

# 避免
try:
    model = load_model(model_path)
except:  # 不要使用裸 except
    pass  # 不要吞掉异常
```

## Commit 信息规范

使用语义化的 Commit 消息：

```
<类型>(<范围>): <简短描述>

<详细描述>

<关闭的 Issue>
```

### 类型

- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档更改
- `style` - 代码格式（不改变功能）
- `refactor` - 代码重构
- `perf` - 性能优化
- `test` - 添加或修改测试
- `chore` - 构建、依赖等杂项更改

### 示例

```
feat(inference): 添加 FP16 精度支持

- 在 Hallo2Pipeline 中添加 dtype 参数
- 自动检测 GPU 内存并选择合适精度
- 减少显存占用 40%

Closes #123
```

## 提交 PR 前的检查清单

- [ ] 代码已使用 Black 格式化
- [ ] 通过了 Flake8 检查
- [ ] 添加或更新了相关文档
- [ ] 添加了测试代码（如果需要）
- [ ] 测试通过
- [ ] Commit 消息清晰和语义化
- [ ] 没有添加不必要的依赖
- [ ] 没有提交日志文件或临时文件

## 审查流程

1. 至少一个维护者会审查你的 PR
2. 可能会要求进行修改
3. 所有反馈解决后，PR 会被合并
4. 感谢你的贡献！

## 问题和讨论

- 如有问题，请在相关 Issue 下讨论
- 对于大的功能改动，建议先开 Issue 讨论
- 欢迎在 Discussions 中分享想法

## 行为准则

我们承诺提供包容、尊重的社区环境。请：

- 尊重不同的观点
- 接受建设性的批评
- 关注对社区最有益的事情
- 对其他社区成员表示同情

不可接受的行为包括：

- 骚扰、辱骂或歧视
- 发表仇恨或冒犯性言论
- 在未同意的情况下发布他人隐私信息

## 许可证

通过提交贡献，你同意你的代码将在 MIT 许可证下发布。

## 需要帮助？

- 查看 [README.md](README.md) 了解基础信息
- 查看 [API 文档](docs/api.md)（如果存在）
- 提出 Issue 询问

感谢你的贡献！🎉
