---
skill_id: review-kb-doc
name: KB 文档审查
description: 用于审查知识库文档的 metadata、层级归属、结构完整性、风险点和建议状态，不自动提升 Legacy 材料。
disable_model_invocation: false
---

# KB 文档审查

## 适用场景

- review 某份标准文档
- review 某份 `master_gdd`
- review 某份 `feature_spec`
- 检查 metadata、层级归属、结构完整性和可批准状态

## 工作方式

### 1. 层级归属是否正确

检查该文档更适合属于：

- `Shared`
- `GameLine`
- `Deployment`
- `Legacy`

### 2. 文档类型是否合理

检查是否更适合：

- `master_gdd`
- `feature_spec`
- 其他知识沉淀文档

### 3. metadata 是否完整

重点检查：

- `doc_id`
- `doc_type`
- `title`
- `hall`
- `game_line`
- `deployment`
- `status`
- `version`
- `source_of_truth`
- `updated_at`
- `summary`

### 4. 结构是否完整

重点检查是否缺少：

- 目标 / 玩家价值
- Scope / Non-scope
- 主流程
- 核心规则
- 依赖
- 边界情况
- Open Questions
- 决策记录
- 变更记录

### 5. 是否混写层级

检查：

- `Shared` 内容是否混入分支事实
- `GameLine` 内容是否混入某个 Deployment 特有规则
- `Legacy` 材料是否被误当成当前事实

## 建议输出

- 审查结论
- 发现的问题
- 缺失项
- 风险点
- 建议动作
- 建议状态

## 注意事项

- 优先指出结构问题、层级问题和事实风险
- 不要陷入措辞微调
- 不要越权拍板
- 不要自动提升 `Legacy` 为事实源
