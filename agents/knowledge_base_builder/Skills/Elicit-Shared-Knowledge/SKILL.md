---
skill_id: elicit-shared-knowledge
name: Shared 层知识抽取
description: 用于梳理 Shared 层公共知识，例如术语、命名规范、设计原则、通用规则和公司级标准。
disable_model_invocation: false
---

# Shared 层知识抽取

## 适用场景

- 梳理公司级术语表
- 梳理设计原则、命名规范、文档规范
- 梳理跨项目通用规则或公共系统定义

## 工作方式

### 1. 先判断 Shared 内容属于哪一类

例如：

- Glossary
- Design Principles
- Naming
- Standards
- Common Systems

### 2. 重点追问

- 统一定义是什么
- 适用范围是什么
- 是否有别名、误叫法、历史叫法
- 是全公司通用，还是只在某个游戏或某个分支成立
- 与哪些相近概念容易混淆

### 3. 明确哪些内容不该放进 Shared

如果某条知识只适用于某个大厅、某个分支、某个功能，就要明确指出它不应落在 `Shared`。

## 建议输出

- 已确认的 Shared 条目
- 不应放入 Shared 的内容
- 待确认项
- 建议落文档

## 注意事项

- `Shared` 层不能混入某个 Deployment 的当前事实
- 不要写具体分支数值或当前版本规则
- 遇到“看似通用，实际是项目习惯”的内容，要单独标记
