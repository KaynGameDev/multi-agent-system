# Skill: Review KB Doc

## 目的

用于审查一份知识库文档是否符合当前 V1 规则。

---

## 适用场景

当用户要求：

- review 某份标准文档
- review 某份 Master GDD
- review 某份 Feature Spec
- 检查 metadata、层级、完整性

---

## 你的检查维度

### 1. 层级归属是否正确
检查该文档更适合属于：
- Shared
- GameLine
- Deployment
- Legacy

### 2. 文档类型是否正确
检查是否更适合：
- `master_gdd`
- `feature_spec`
- `GameLineOverview`

### 3. metadata 是否完整
重点检查：
- `doc_id`
- `doc_type`
- `title`
- `project`
- `hall`
- `game_line`
- `deployment`
- `status`
- `version`
- `source_of_truth`
- `updated_at`
- `summary`

### 4. 结构是否完整
例如 Feature Spec 是否缺：
- 目标 / 玩家价值
- Scope / Non-scope
- 规则
- 流程
- 依赖
- 边界情况
- 开放问题
- 变更与决策记录

### 5. 知识层是否混写
检查：
- Shared 内容是否混入分支事实
- GameLine 内容是否混入某个 Deployment 的特有规则
- Legacy 内容是否被当成当前事实

### 6. 可批准性判断
判断建议状态更接近：
- `draft`
- `in_review`
- `approved`

---

## 输出格式

### 审查结论
### 发现的问题
### 缺失项
### 风险点
### 建议动作
### 建议状态

---

## 注意事项

- 优先指出结构问题和层级问题
- 不要陷入措辞细节
- 不要越权拍板