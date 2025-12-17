<!-- GPU/CPU监测和预设系统调查报告 -->

### Code Sections (The Evidence)

#### GPU/CPU监测系统
- `backend/app/services/hardware_service.py:74-126` (_detect_gpu): GPU核心检测逻辑，包含CUDA可用性检测、GPU数量、显存容量和GPU型号获取
- `backend/app/services/hardware_service.py:128-189` (_detect_cpu): CPU信息检测，包含物理/逻辑核心数、CPU型号和最大频率
- `backend/app/services/hardware_service.py:191-217` (_detect_memory): 系统内存检测，包含总内存和可用内存
- `backend/app/models/hardware_models.py:9-60` (HardwareInfo): 硬件信息数据模型，定义了GPU、CPU、内存的结构化数据
- `backend/app/models/hardware_models.py:63-108` (OptimizationConfig): 基于硬件的优化配置模型，包含批处理大小、并发数、设备推荐等
- `backend/app/main.py:333-350` (get_hardware_basic): 硬件信息API端点，提供基础的硬件状态查询

#### 预设系统架构
- `backend/app/models/job_models.py:50-74` (SenseVoiceSettings): SenseVoice配置数据模型，定义了预设ID、增强模式、校对模式、翻译模式等
- `frontend/src/components/editor/PresetSelector.vue:98-153` (presets): 前端预设配置数组，定义了6种预设方案的完整配置
- `llmdoc/architecture/sensevoice-presets.md:29-66` (预设应用流程): 预设系统架构文档，详细描述了预设配置和执行流程
- `backend/app/services/transcription_service.py`: 转录服务，负责应用预设配置到实际转录流程
- `backend/app/api/routes/transcription_routes.py:32-41` (SenseVoiceSettingsAPI): API请求模型，处理前端预设配置请求

#### GPU内存管理
- `backend/app/services/demucs_service.py:381-382` (torch.cuda.empty_cache): Demucs服务中的GPU内存清理
- `backend/app/services/demucs_service.py:417-418` (torch.cuda.empty_cache): Demucs模型卸载时的内存清理
- `backend/app/services/job_queue_service.py:412-413` (CUDA显存清理): 任务队列服务中的显存管理逻辑

### Report (The Answers)

#### result

**GPU/CPU监测现状：**
- ✅ **基础硬件检测完备**: 项目具备完整的硬件检测系统，可识别GPU数量、显存容量、CPU核心数、内存大小等关键信息
- ✅ **静态检测机制**: 通过`CoreHardwareDetector`类提供系统启动时的硬件信息扫描
- ✅ **优化配置生成**: `CoreOptimizer`根据硬件配置自动推荐最优的批处理大小、并发数和设备选择
- ❌ **缺失实时监测**: 没有实时的GPU显存使用率、CPU利用率、内存使用情况动态监测
- ❌ **缺失OOM保护**: 未发现显式的Out of Memory预防和保护机制
- ❌ **缺失资源告警**: 没有资源使用过量的预警和自动处理机制

**预设系统架构：**
- ✅ **6层预设方案**: 提供从"极速"到"重译"的渐进式预设选择，满足不同精度需求
- ✅ **模块化配置**: 预设由增强模式、校对模式、翻译模式三个维度组合而成
- ✅ **前后端分离**: 前端`PresetSelector.vue`负责用户界面，后端`SenseVoiceSettings`负责数据模型
- ✅ **API集成完善**: 通过`/api/start`端点接收预设配置并应用到转录服务
- ❌ **硬件自适应缺失**: 预设选择没有根据实际硬件能力进行智能推荐
- ❌ **动态预设不足**: 缺少基于运行时性能反馈的预设自动调整

#### conclusions

1. **硬件检测基础扎实**: 项目已具备完整的静态硬件检测能力，为性能优化提供数据基础
2. **预设系统设计成熟**: 6种预设方案覆盖了从速度到精度的完整应用场景
3. **实时监测能力缺失**: 当前无法动态监测GPU显存使用率和系统资源状态
4. **智能化程度有限**: 预设选择和硬件配置缺乏自动化和智能化机制
5. **资源管理待完善**: GPU内存管理仅有基本的清理机制，缺乏精细化的资源调度

#### relations

- `hardware_service.py` → `hardware_models.py`: 硬件检测服务生成硬件信息数据模型
- `hardware_models.py` → `transcription_service.py`: 硬件信息影响转录服务的设备选择和参数配置
- `PresetSelector.vue` → `SenseVoiceSettingsAPI`: 前端预设选择通过API转换为后端配置模型
- `SenseVoiceSettings` → `transcription_service.py`: 预设配置驱动转录服务的具体执行策略
- `demucs_service.py` ← `job_queue_service.py`: 任务队列管理Demucs服务的资源清理和内存管理

### 需要新增的功能

#### GPU/CPU监测增强
1. **实时资源监测服务**:
   - GPU显存使用率实时监测
   - CPU利用率和温度监测
   - 系统内存使用率追踪
   - 磁盘I/O性能监控

2. **OOM保护机制**:
   - 预测性内存分配检查
   - 动态批处理大小调整
   - 内存不足时的优雅降级
   - 显存碎片自动整理

3. **资源告警系统**:
   - 资源使用阈值预警
   - 性能瓶颈自动识别
   - 异常情况自动恢复
   - 资源状态实时推送

#### 预设系统智能化
1. **硬件自适应预设**:
   - 根据GPU显存自动推荐合适预设
   - 基于CPU性能调整并发参数
   - 内存限制下的预设自动降级
   - 个性化预设学习推荐

2. **动态预设调整**:
   - 运行时性能反馈收集
   - 基于准确率自动调优参数
   - 用户满意度驱动的预设优化
   - A/B测试的预设效果评估

3. **预设管理界面**:
   - 自定义预设创建和保存
   - 预设效果对比和分享
   - 预设使用统计和分析
   - 场景化预设模板库

### 改进建议

#### 短期改进 (1-2周)
1. **增加实时监测API**: 在`system_routes.py`中添加`/api/system/monitor`端点，提供实时的资源使用情况
2. **完善GPU内存管理**: 在转录服务中添加显存使用量监测和预警机制
3. **预设硬件适配**: 在`PresetSelector.vue`中根据硬件信息智能推荐合适的预设方案

#### 中期改进 (1-2月)
1. **构建资源监测服务**: 创建独立的`ResourceMonitorService`，提供系统资源实时监测和历史数据分析
2. **智能预设系统**: 基于机器学习算法，根据硬件配置和用户习惯自动优化预设参数
3. **性能优化引擎**: 集成硬件检测、预设选择、资源调度的一体化性能优化系统

#### 长期规划 (3-6月)
1. **自适应AI优化**: 使用强化学习技术，让系统根据实际运行效果自动调整所有参数
2. **云原生资源管理**: 支持分布式环境下的资源调度和负载均衡
3. **全链路性能监控**: 从视频输入到字幕输出的完整性能分析和优化建议系统