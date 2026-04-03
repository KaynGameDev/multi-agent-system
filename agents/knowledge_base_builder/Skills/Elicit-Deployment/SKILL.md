---
skill_id: elicit-deployment
name: Deployment 层知识抽取
description: 用于梳理某个 Deployment 的当前事实、与 GameLine 共性的差异，以及该分支独有的规则、功能和限制。
disable_model_invocation: false
---

# Deployment 层知识抽取

## 适用场景

- 梳理某个大厅或包体下某个游戏分支的当前事实
- 梳理分支版 Master GDD
- 梳理分支特有功能、限制、运营规则
- 梳理该分支与 GameLine 共性的差异

## 工作方式

### 1. 必须先锁定目标

至少明确：

- hall
- game_line
- deployment

### 2. 重点追问

- 这个分支当前实际怎么做
- 它相对 GameLine 共性改了哪些地方
- 哪些功能、规则、界面、数值或运营点是该分支独有的
- 当前版本范围是什么
- 哪些说法已经确认，哪些还只是印象

### 3. 强制和母体区分

每轮都要检查：

- 这是 `GameLine` 共性还是 `Deployment` 特例
- 这是当前事实还是历史遗留
- 这是正式规则还是曾经讨论过的方案

## 建议输出

- 已确认的 Deployment 事实
- 相对 GameLine 的差异
- 疑似历史遗留或待确认内容
- 建议落文档

## 注意事项

- `Deployment` 层回答的是“这个分支当前实际怎么做”
- 不要把其他大厅的事实混进来
- 同名游戏跨大厅不能合并成一个事实源
