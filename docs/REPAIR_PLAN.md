# 视频转字幕系统 - 5大问题修复方案

## 问题分析总结

### 问题1：幽灵任务 - 任务已删除但依然显示
**根本原因**：
- 前端任务列表完全依赖 localStorage 持久化
- 删除任务时只从前端删除，未检查后端实际状态
- 应用启动时未调用后端 `getIncompleteJobs` 同步实际存在的任务
- 前端 localStorage 成为唯一的真实源，与后端 jobs 目录不同步

---

## 修改方案

### 方案1：修复幽灵任务问题

#### 后端改动：
1. **增强 `/api/incomplete-jobs` 接口**（transcription_routes.py）
   - 改进为 `/api/sync-tasks` 端点
   - 返回所有任务（处理中 + 已完成）及其详细状态
   - 包含 job_id, filename, status, progress, created_time 等
   - 同时清理 job_index.json 中的无效映射（文件已删除的)

2. **改进任务��除逻辑**（transcription_service.py）
   - 删除任务时主动更新 job_index.json（移除映射）
   - 确保 jobs 目录与索引文件一致

#### 前端改动：

1. **修改应用启动流程**（App.vue）
  
   ```
   应用启动 → 从后端同步任务列表 → 更新 localStorage
             ↓
          订阅全局 SSE 事件 → 实时更新任务状态
   ```
   - 替代当前的 localStorage 恢复机制
   - 使用后端数据作为真实源
   
2. **改进 unifiedTaskStore**（unifiedTaskStore.js）
   - 新增 `syncTasksFromBackend()` 方法
   - 应用启动时调用此方法，将后端任务同步到 store
   - 删除任务时，不仅从前端删除，还要从 localStorage 清理

3. **处理 404 错误**（transcriptionApi.js / EditorView.vue）
  
   - 任何 API 返回 404 时，自动从前端删除该任务
   - 刷新任务列表

---

### 方案2：修复任务打开 404 错误

#### 问题根源：
- job_index.json 存储的文件路径映射不完整或过期
- 前端 localStorage 包含已删除的 job_id，但后端找不到

#### 改动方案：

1. **后端清理机制**（新增模块）
   - 启动时运行一次 jobs 目录完整扫描
   - 对于 job_index.json 中的映射，验证：
     - jobs/{job_id}/ 目录是否存在
     - 对应的文件是否存在
   - 删除无效映射
   - 定期清理机制（可选）

2. **后端 `/api/status/{job_id}` 改进**（transcription_routes.py）
   - 优先查询内存队列
   - 次优查询 job_index.json
   - 都不存在时返回 404 + 清理信息
   - 改进错误响应，告知前端该任务不存在

3. **前端错误处理**（EditorView.vue, TaskListView.vue）
   - 当收到 404 时，自动删除本地 localStorage 中的该任务
   - 显示友好错误提示

---

### 方案3：实现任务卡片实时更新

#### 改动方案：

1. **后端 SSE 初始状态优化**（sse_service.py）
   - `get_initial_state()` 返回所有任务（历史完成 + 处理中）
   - 返回字段：job_id, filename, status, progress, created_time
   - 避免只返回活跃任务导致的完成任务不更新

2. **后端缩略图生成**（新增功能）
   - 在视频处理时提取第一帧作为缩略图
   - 存储为 `jobs/{job_id}/thumbnail.jpg`
   - API 端点 `/api/media/thumbnail/{job_id}` 提供缩略图
   - 或在任务状态中包含 Base64 编码的缩略图

3. **前端订阅更新**（TaskListView.vue）
   - 挂载时订阅全局 SSE 事件
   - 监听 `job_progress`, `job_status`, `job_completed` 事件
   - 实时更新对应任务的：进度条、状态文本、缩略图

4. **前端数据绑定**（TaskListView.vue）
   - 任务标题 → 从 task.filename 或新增的 task.title
   - 进度条 → 从 task.progress
   - 缩略图 → 从 `/api/media/thumbnail/{job_id}` 或 task.thumbnail_url
   - 时间戳 → 从 task.created_time，格式化为 YYYY-MM-DD HH:mm

---

### 方案4：实现任务重命名功能

#### 改动方案：

1. **后端数据模型**（job_models.py）
   - 在 `JobState` 增加 `title` 字段（可选，默认为 filename）
   ```python
   @dataclass
   class JobState:
       ...
       title: str = ""  # 用户自定义的任务名称
   ```
   - 保存到任务状态文件中

2. **后端 API**（transcription_routes.py）
   - 新增 `POST /api/rename-job/{job_id}` 端点
   - 请求体：`{"title": "新名���"}`
   - 更新任务的 title 字段
   - 保存到 jobs/{job_id}/state.json 或任务配置文件

3. **前端 UI**（TaskListView.vue）
  
   - 任务卡片上的"编辑"按钮改为"重命名"
   - 点击弹出简单的对话框或行内编辑
   - 输入框显示当前名称
   - 保存时调用后端 `/api/rename-job/{job_id}` API
   - 成功后更新本地 store
   
4. **前端显示**（TaskListView.vue）
  
   - 任务标题优先显示 task.title（如果有）
   - 否则显示 task.filename
   - 可在标题下显示原文件名作为副标题

---

### 方案5：实现文件上传双模式

#### 改动方案：
1. **前端 UI 重构**（新增 UploadView 或改进 TaskListView）
   - 添加两个选项卡：
     - **选项卡A：直接上传** - 用户上传文件，系统保存到 input 目录
     - **选项卡B：从本地目录选择** - 扫描 input 目录已有的文件，用户可多选

2. **后端 API**（file_routes.py）
   - 改进 `/api/files` 接口，返回 input 目录所有文件列表
   - 返回格式：`[{filename, size, created_time}, ...]`
   - 支持分页和过滤（可选）

3. **上传流程 A：直接上传（现有逻辑改进）**
   - 用户选择文件 → 上传到后端
   - 后端接收：`/api/upload`
   - 保存到：`input/{filename}`
   - 自动创建任务（无需用户手动创建）
   - 自动加入队列
   - 返回新创建的 job_id

4. **选择流程 B：从 input 目录选择**
   - 用户点击"浏览 input 目录"
   - 前端调用 `/api/files` 获取列表
   - 用户多选文件
   - 对每个选中文件调用 `/api/create-job?filename={filename}`
   - 后端使用已有的文件创建任务（不复制）
   - 任务创建后自动加入队列或等待用户启动

5. **关键设计要点**
   - **流程 A**：上传文件到 input → 自动创建任务并加入队列
   - **流程 B**：input 目录已有文件 → 选择创建任务，无需复制
   - 不要在流程 B 中复制文件（input 已有，无需重复存储）

6. **前端文件列表 UI**（新组件）
   - 展示 input 目录所有视频文件
   - 显示文件大小、修改时间
   - Checkbox 多选
   - 批量创建任务按钮

---

## 实施步骤顺序

### 第一阶段：修复数据同步（高优先级）

1. ✓ 后端增强 `/api/sync-tasks` 接口
2. ✓ 后端删除任务时清理 job_index.json
3. ✓ 前端 App.vue 启动时调用 sync-tasks
4. ✓ 前端处理 404 错误，清理无效任务
5. ✓ 后端启动时清理无效映射

### 第二阶段：实时更新（中等优先级）

6. ✓ 后端优化 SSE initial_state（返回所有任务）
7. ✓ 后端生成并提供缩略图 API
8. ✓ 前端订阅 SSE 并更新卡片
9. ✓ 前端集成缩略图显示（TaskListView 任务卡片）
10. ✓ 修改时间戳显示格式为 YYYY-MM-DD HH:mm

### 第三阶段：功能增强（中等优先级）

11. ☐ 后端添加 title 字段和重命名 API
12. ☐ 前端 UI 改为重命名功能

### 第四阶段：上传优化（低优先级）

13. ☐ 后端改进 `/api/files` 接口
14. ☐ 前端实现双模式上传 UI
15. ☐ 前端处理多选和批量创建

---

## 关键技术要点

1. **前后端同步策略**：
   - 后端 jobs 目录为真实源
   - 前端 localStorage 为缓存
   - 定期（启动、SSE 初始化）从后端同步

2. **错误恢复**：
   - 404 → 删除本地无效任务
   - job_index.json → 启动时清理

3. **实时更新**：
   - SSE 包含所有任务状态变更
   - 前端订阅并响应更新

4. **文件存储**：
   - input/ → 源文件存储
   - jobs/{job_id}/ → 工作目录和中间结果
   - output/ → 导出位置（可选）

---

## 修改文件清单

**后端文件**：

- [ ] `/backend/app/api/routes/transcription_routes.py` - 新增/改进 API
- [ ] `/backend/app/services/transcription_service.py` - 删除时清理索引
- [ ] `/backend/app/services/sse_service.py` - 优化 initial_state
- [ ] `/backend/app/models/job_models.py` - 添加 title 字段
- [ ] `/backend/app/services/job_index_service.py` - ���理和验证函数
- [ ] `/backend/app/main.py` - 启动时清理无效任务

**前端文件**：
- [ ] `/frontend/src/App.vue` - 启动时同步任务
- [ ] `/frontend/src/views/TaskListView.vue` - 重命名 UI + SSE 订阅
- [ ] `/frontend/src/stores/unifiedTaskStore.js` - 同步方法 + 404 处理
- [ ] `/frontend/src/services/api/transcriptionApi.js` - 新增 API 方法
- [ ] `/frontend/src/components/FileUploadPanel.vue` (新增) - 双模式上传组件

---

## 测试检查清单

1. [ ] 删除任务后刷新页面，任务不再显示
2. [ ] 历史完成任务仍能打开编辑器
3. [ ] 任务卡片实时显示进度和状态
4. [ ] 任务缩略图正常显示
5. [ ] 点击重命名可修改任务名称
6. [ ] 直接上传文件能正常处理
7. [ ] 从 input 目录选���文件能多选
8. [ ] 时间戳格式为 YYYY-MM-DD HH:mm
