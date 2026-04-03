---
name: Feature Spec 抽取
description: 用于围绕单个功能逐步抽取知识，并整理成 feature_spec 草稿骨架，包含决策记录和变更记录。
disable_model_invocation: false
---

# Feature Spec 抽取

## 适用场景

- 梳理单个功能
- 梳理子系统、玩法模块或关键流程
- 把讨论内容整理成 `feature_spec` 草稿骨架

## 工作方式

### 1. 先锁定 Feature 范围

先明确：

- 功能名称
- 所属 GameLine 或 Deployment
- 是母体共性还是分支特例

### 2. 建议追问顺序

1. 功能目标是什么
2. 给玩家带来的价值是什么
3. Scope / Non-scope 是什么
4. 主流程是什么
5. 核心规则是什么
6. 有哪些状态变化
7. 参数、公式、数据依赖是什么
8. 边界情况和异常情况是什么
9. 依赖哪些其他系统
10. 最近有哪些关键变更与决策

### 3. 每轮都整理成骨架

至少沉淀：

- 已确认内容
- 待确认内容
- 缺失章节提醒
- 当前可落下的 `feature_spec` 骨架

## 注意事项

- 不要直接写成一大段说明文
- Feature Spec 必须包含决策记录与变更记录
- 遇到分支差异时，要明确是否只适用于当前 Deployment
- 不引入 ADR
