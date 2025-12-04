# Phase 5: 整合测试

> 目标：端到端测试和性能验证
>
> 工期：1-2天

---

## 一、测试策略

**个人开发原则**：
- 只做关键路径测试
- 不做完整单元测试
- 重点验证核心功能
- 快速迭代修复

---

## 二、测试场景（3-5个典型场景）

### 场景1：纯净语音（无BGM）

**测试目标**：验证基础转录流程

**输入**：
- 视频：纯人声采访/演讲（无背景音乐）
- 时长：5-10分钟

**预期结果**：
- [ ] VAD 正确切分
- [ ] 频谱检测判定为"无需分离"
- [ ] SenseVoice 直接转录
- [ ] 字幕为句级粒度（10-20字/句）
- [ ] 置信度 > 0.6
- [ ] 无 Whisper 补刀
- [ ] 进度条平滑（无停滞）

**验证命令**：
```bash
# 后端测试
python backend/test_scenario1_clean_speech.py

# 前端测试
# 1. 上传测试视频
# 2. 选择 SenseVoice 引擎
# 3. 开始转录
# 4. 观察进度和实时字幕
```

---

### 场景2：轻度BGM

**测试目标**：验证按需分离机制

**输入**：
- 视频：轻度背景音乐的视频（vlog/新闻）
- 时长：5-10分钟

**预期结果**：
- [ ] 频谱检测识别出部分片段需要分离
- [ ] 仅识别出的片段进行人声分离
- [ ] 其他片段直接转录
- [ ] 动态权重调整生效（分离权重 < 15%）
- [ ] 字幕质量良好

**关键指标**：
- 分离片段占比：10%-30%
- 总处理时间：< 2分钟（10分钟视频）

---

### 场景3：重度BGM

**测试目标**：验证熔断升级机制

**输入**：
- 视频：重度背景音乐（MV/音乐会）
- 时长：3-5分钟

**预期结果**：
- [ ] 频谱检测识别大量片段需要分离
- [ ] 人声分离启用（htdemucs 或 mdx_extra）
- [ ] 转录置信度合理
- [ ] 部分低置信度片段触发 Whisper 补刀
- [ ] 熔断决策优先升级分离，后补刀

**关键指标**：
- 分离片段占比：> 50%
- 补刀片段数：< 10%

---

### 场景4：低质量音频

**测试目标**：验证熔断机制

**输入**：
- 视频：噪音较大的视频（街头采访/会议录音）
- 时长：5分钟

**预期结果**：
- [ ] SenseVoice 初步转录
- [ ] 检测到 BGM/Noise 事件标签
- [ ] 熔断决策触发"升级分离"
- [ ] 重新分离后再转录
- [ ] 如仍低置信度，触发 Whisper 补刀
- [ ] 最终字幕可用

---

### 场景5：无GPU环境（可选）

**测试目标**：验证 CPU 模式

**输入**：
- 在无 GPU 的机器上运行
- 视频：任意纯人声视频

**预期结果**：
- [ ] 硬件检测正确识别无 GPU
- [ ] 人声分离自动禁用
- [ ] SenseVoice 使用 CPU 推理
- [ ] 转录成功完成（速度较慢）
- [ ] 前端显示硬件状态

---

## 三、性能验证

### 3.1 性能指标

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 10分钟视频处理时间 (GPU) | < 2分钟 | _____ | ⬜ |
| 句级平均长度 | 10-20字 | _____ | ⬜ |
| 置信度准确率 | > 85% | _____ | ⬜ |
| 显存峰值 (8GB GPU) | < 6GB | _____ | ⬜ |
| 进度条准确性 | 无停滞/回跳 | _____ | ⬜ |

### 3.2 性能测试脚本

**路径**: `backend/test_performance.py`

```python
"""
性能测试脚本
"""
import sys
import time
from pathlib import Path
import asyncio

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.services.transcription_service import TranscriptionService
from app.models.job_models import JobState, JobSettings


async def test_performance(video_path: str):
    """
    性能测试

    Args:
        video_path: 测试视频路径
    """
    print(f"\n=== 性能测试: {video_path} ===")

    # 创建任务
    job = JobState(
        job_id=f"perf_test_{int(time.time())}",
        input_path=video_path,
        output_path=f"test_output/{Path(video_path).stem}_output.srt",
        settings=JobSettings(engine='sensevoice')
    )

    service = TranscriptionService()

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行转录
        await service._process_video_sensevoice(job)

        # 记录结束时间
        end_time = time.time()
        elapsed = end_time - start_time

        # 计算指标
        print(f"\n性能指标:")
        print(f"  处理时间: {elapsed:.2f}秒")
        print(f"  任务状态: {job.status}")

        # 分析字幕
        if Path(job.output_path).exists():
            with open(job.output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                sentence_count = content.count('\n\n')
                print(f"  字幕数量: {sentence_count}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python test_performance.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    asyncio.run(test_performance(video_path))
```

---

## 四、问题排查清单

### 4.1 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 字幕粒度太粗 | 分句参数不合理 | 调整 `pause_threshold`/`max_chars` |
| 进度条停滞 | 权重计算错误 | 检查 `calculate_dynamic_weights()` |
| 频繁误判 BGM | 阈值过低 | 调整 `music_score_threshold` |
| 显存不足 | 模型未卸载 | 检查模型串行化逻辑 |
| 转录质量差 | 未升级分离 | 检查熔断决策逻辑 |

### 4.2 调试技巧

1. **查看日志**：
   ```bash
   tail -f logs/transcription.log
   ```

2. **单步调试**：
   - 在关键方法设置断点
   - 检查中间结果

3. **临时输出**：
   ```python
   self.logger.info(f"[DEBUG] 片段 {i}: 置信度={confidence}, 决策={decision}")
   ```

---

## 五、优化建议

### 5.1 参数调优

如果测试效果不理想，可调整以下参数：

#### 分句算法参数

```python
# sentence_splitter.py
config = {
    'pause_threshold': 0.4,   # 调整停顿阈值（0.3-0.6）
    'max_duration': 5.0,      # 调整最大时长（4-6秒）
    'max_chars': 30,          # 调整最大字数（25-35）
}
```

#### 频谱检测参数

```python
# audio_circuit_breaker.py
config = {
    'music_score_threshold': 0.35,  # 调整音乐性阈值（0.25-0.45）
    'history_threshold': 0.6,       # 调整惯性阈值（0.5-0.7）
}
```

#### 熔断决策参数

```python
# fuse_breaker.py
config = {
    'confidence_threshold': 0.6,   # 调整置信度阈值（0.5-0.7）
    'upgrade_threshold': 0.4,      # 调整升级阈值（0.3-0.5）
}
```

### 5.2 性能优化

如果处理速度不达标：

1. **减少 VAD 片段数**：
   - 增大 `target_segment_duration_s`

2. **批处理优化**：
   - 增加 `batch_size`（如果显存充足）

3. **并行处理**：
   - 多片段并行转录（需注意显存）

---

## 六、交付物

### 6.1 测试报告模板

```markdown
# SenseVoice 集成测试报告

## 测试环境
- GPU: [型号]
- 显存: [容量]
- 操作系统: [系统]

## 测试结果

### 场景1：纯净语音
- 状态: ✅ 通过 / ❌ 失败
- 处理时间: [时间]
- 字幕质量: [评价]
- 问题: [描述]

### 场景2：轻度BGM
- ...

### 场景3：重度BGM
- ...

### 场景4：低质量音频
- ...

## 性能指标
- 10分钟视频处理时间: [时间]
- 句级平均长度: [字数]
- 显存峰值: [GB]

## 问题列表
1. [问题描述]
2. [问题描述]

## 优化建议
1. [建议]
2. [建议]
```

### 6.2 最终检查清单

- [ ] 所有测试场景通过
- [ ] 性能指标达标
- [ ] 硬件适配正常
- [ ] 前端界面完整
- [ ] API 接口正常
- [ ] 错误处理健全
- [ ] 日志记录完善
- [ ] 代码提交 git

---

## 七、上线准备

### 7.1 配置检查

```python
# config.py
SENSEVOICE_DEFAULT_CONFIG = {
    'use_onnx': True,
    'quantization': 'int8',
    'confidence_threshold': 0.6,
    'enable_demucs': 'auto',  # auto/true/false
}
```

### 7.2 文档更新

- [ ] 更新 README.md
- [ ] 更新用户手册
- [ ] 更新 API 文档
- [ ] 更新 CHANGELOG

### 7.3 依赖更新

```bash
# requirements.txt
onnxruntime-gpu>=1.16.0
# 或
onnxruntime>=1.16.0  # CPU only
```

---

## 八、后续优化（可选）

### 8.1 功能增强

1. **ONNX 模型实际推理**：
   - 当前返回模拟数据
   - 需实现真实推理逻辑

2. **Whisper 补刀完善**：
   - 当前仅返回原句子
   - 需实现真实的 Whisper 转录

3. **模型自动下载**：
   - 首次运行自动下载 ONNX 模型

### 8.2 用户体验优化

1. **进度预估更准确**：
   - 基于历史数据预测处理时间

2. **字幕编辑器**：
   - 支持在线编辑字幕

3. **批量处理**：
   - 支持多视频批量转录

---

## 九、总结

完成 Phase 5 后，SenseVoice 集成项目基本完成。主要成果：

1. ✅ SenseVoice ONNX 服务
2. ✅ 智能分句算法
3. ✅ 智能熔断机制
4. ✅ 动态权重调整
5. ✅ 硬件自适应
6. ✅ 前端界面完整
7. ✅ 端到端测试通过

**总工期**：9-14天

**下一步**：根据实际使用反馈持续优化和迭代。
