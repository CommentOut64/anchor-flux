<!-- FuseBreakerV2 实现状态调查报告 -->

### Code Sections (The Evidence)

- `backend/app/services/fuse_breaker_v2.py` (FuseBreakerV2 class): 熔断决策器V2增强版，负责监控SenseVoice转录质量并决定是否升级分离模型。
- `backend/app/models/circuit_breaker_models.py` (SeparationLevel enum): 定义三个分离级别（NONE、HTDEMUCS、MDX_EXTRA）及升级路径。
- `backend/app/models/circuit_breaker_models.py` (FuseDecision dataclass): 熔断决策结果数据结构，包含action、target_level、reason字段。
- `backend/app/services/audio/chunk_engine.py` (AudioChunk dataclass): 音频片段数据结构，包含分离状态、熔断重试计数等字段。
- `backend/app/models/job_models.py` (PreprocessingConfig dataclass): 预处理配置，包含熔断相关参数（enable_fuse_breaker、fuse_max_retry、fuse_confidence_threshold、fuse_auto_upgrade）。
- `backend/app/services/fuse_breaker.py` (FuseBreaker class): 旧版熔断决策器，用于对比参考。
- `backend/app/services/monitoring/hardware_monitor.py` (HardwareMonitor class): 硬件监测服务，提供GPU/CPU/内存监测功能。

### Report (The Answers)

#### result

**1. allow_mdx_extra 配置参数实现状态：未实现**

- FuseBreakerV2中不存在`allow_mdx_extra`配置参数
- MDX_EXTRA升级由`auto_upgrade`参数控制（第二次重试时自动升级到MDX_EXTRA）
- 升级路径硬编码：第一次重试NONE→HTDEMUCS，第二次重试HTDEMUCS→MDX_EXTRA（当auto_upgrade=True时）
- 无条件限制MDX_EXTRA升级的机制

**2. weight_strategy 多标签加权策略实现状态：部分实现**

- 实现了单标签权重机制（event_weights字典）：BGM=1.0、Music=0.9、Noise=0.8、Applause=0.6
- 加权置信度计算公式：`weighted_confidence = confidence * (1 + weight)`
- 仅支持单个event_tag，不支持多标签列表
- 不存在"max"或"sum"策略选择，使用固定的乘法加权方式

**3. 多标签处理逻辑实现状态：未实现**

- AudioChunk.spectrum_diagnosis字段存在但未在FuseBreakerV2中使用
- should_fuse()方法接收单个event_tag参数（字符串），不支持event_tags列表
- 第81行：`event_tag = sv_result.get('event_tag')`获取单个标签
- 第98行：检查`if not event_tag or event_tag not in self.event_weights`，仅处理单标签
- 无多标签聚合逻辑

**4. _get_next_level()方法allow_mdx_extra限制实现状态：未实现**

- _get_next_level()方法（第128-154行）不检查任何allow_mdx_extra参数
- 升级逻辑完全由auto_upgrade和fuse_retry_count控制
- 第150-151行：当auto_upgrade=True且fuse_retry_count==1时，直接返回MDX_EXTRA
- 无条件检查或限制MDX_EXTRA升级的代码

**5. FuseMetrics监控指标集成实现状态：未实现**

- FuseBreakerV2中不存在FuseMetrics类或监控指标集成
- 仅有get_statistics()方法（第219-231行）返回配置参数，不返回运行时指标
- 无升级次数、触发次数、成功率等监控数据收集
- HardwareMonitor存在但与FuseBreakerV2无集成关系
- 无事件标签触发统计、加权置信度分布等指标

#### conclusions

- **当前实现是简化版本**：FuseBreakerV2采用单标签、固定权重、自动升级的设计，不支持复杂的多标签加权策略
- **MDX_EXTRA升级由auto_upgrade控制**：第二次重试时自动升级到最高级别，无条件限制机制
- **缺少监控指标**：没有运行时统计数据收集，仅保存配置参数
- **配置参数在PreprocessingConfig中定义**：enable_fuse_breaker、fuse_max_retry、fuse_confidence_threshold、fuse_auto_upgrade四个参数
- **升级路径固定**：NONE→HTDEMUCS→MDX_EXTRA，无灵活配置选项
- **事件标签权重固定**：BGM/Music/Noise/Applause四个标签，权重在__init__中硬编码

#### relations

- `FuseBreakerV2`在should_fuse()方法中使用`AudioChunk`对象和SenseVoice转录结果判断是否需要升级
- `FuseBreakerV2._get_next_level()`调用`SeparationLevel.next_level()`获取下一个升级级别
- `FuseBreakerV2.execute_upgrade()`调用`DemucsService.separate_chunk()`执行实际分离
- `PreprocessingConfig`中的熔断参数在`JobSettings`中被使用，通过`ConfigAdapter`传递给流水线
- `FuseDecision`返回值包含`FuseAction`枚举和`SeparationLevel`目标级别
- `AudioChunk`的fuse_retry_count字段在_get_next_level()中被检查以决定升级策略
- `HardwareMonitor`与FuseBreakerV2无直接关系，属于独立的硬件监测系统
