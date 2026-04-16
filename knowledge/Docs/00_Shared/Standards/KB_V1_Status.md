# KB V1 状态

## 说明

这份文档用于记录 KB V1 的执行状态。

当前仓库只能确认知识库结构、模板和规范约束，不能单凭仓库内容直接断定真实推进阶段。
因此，凡是仓库中无法证明的推进信息，都应保持为待确认，不要臆造。

## 当前判断

- 当前官方 milestone 名称：待确认
- 基于仓库证据的阶段判断：基础能力已落地，内容迁移与审查仍处早期
- 当前 owner：待确认
- 当前主 blocker：缺少已审查的 canonical 内容、缺少统一状态登记、关键业务映射信息仍未补齐

## 从仓库可确认的前提

- KB V1 只覆盖设计知识
- Canonical 格式统一为 Markdown
- V1 不引入 ADR
- 核心文档类型保留 `master_gdd` 与 `feature_spec`
- 知识库按 `Shared`、`GameLine`、`Deployment`、`Legacy` 四层组织
- `Legacy` 默认不是事实源，不能自动升格

## 仓库快照（2026-04-15）

- `knowledge/Docs` 当前共有 10 份 Markdown 文档
- 已落地内容主要集中在 `00_Shared/Standards`、`30_Review` 与 `50_Templates`
- `10_GameLines`、`20_Deployments`、`40_Legacy` 当前仍没有已填充的 canonical 文档
- 当前已有 3 份模板：`TEMPLATE_MASTER_GDD.md`、`TEMPLATE_FEATURE_SPEC.md`、`TEMPLATE_GAME_LINE_OVERVIEW.md`

## 当前阶段

- 非官方命名：平台/运行时能力基本就绪，知识内容迁移与 review 仍在早期阶段

## 本阶段目标

- 把“有结构、能写入、能检索”的基础能力，推进到“有首批已审查事实文档”
- 建立可持续更新的 status / review / migration 记录口径
- 产出至少一批可作为事实源的 `GameLine` 与 `Deployment` 文档

## 已完成

- 已建立基础目录结构
- 已存在共享标准目录
- 已提供 `master_gdd`、`feature_spec`、`game_line_overview` 模板入口
- 已存在 KB Builder 专用提示词与技能入口
- 已落地 `knowledge_base_builder_agent`，支持知识抽取、文档落点判断与 KB V1 状态跟踪
- 已落地 KB 读取工具与 builder 写入工具，KB 写入带显式确认门槛
- Jade runtime 已切到单一生产路由故事：单回合单 worker、`pending_action` 优先、解析器合同路由、显式 fallback
- 本地验证：`PYTHONPATH=. pytest -q --ignore=tests/test_durable_resume.py` 通过（151 passed, 10 subtests passed）

## 进行中

- 补齐 `00_Shared` 之外的首批 canonical 内容
- 梳理射击塔防组产品线事实，包括 GameLine 差异、Deployment 差异、游戏与平台映射矩阵
- 把 `Decision_Backlog`、`Migration_Checker`、`Open_Questions` 从占位文件补成可追踪清单
- 让测试与运行环境达到更稳定的一键验证状态

## 阻塞项

- 缺少可证明当前 milestone 的状态记录
- 缺少已完成项与待办项的统一更新口径
- 缺少对已审查文档范围的明确登记
- `10_GameLines`、`20_Deployments`、`40_Legacy` 还没有已填充的 canonical 文档
- 产品线关键事实仍未补齐：泰国副包新架构说明、四款捕鱼玩法差异、印尼主包 vs 副包差异、游戏与平台映射矩阵
- 默认 `pytest -q` 不是一键绿：需要 `PYTHONPATH=.`，且 `tests/test_durable_resume.py` 依赖当前环境中不可用的 `langgraph.checkpoint.sqlite`

## 下一步

- 明确官方 owner、milestone 名称、更新频率，并在本文件中持续登记
- 优先补出 1 份 `GameLine Overview` 与 1 份 `Deployment` 级 `master_gdd` / `feature_spec`
- 回填 `Migration_Checker`、`Decision_Backlog`、`Open_Questions` 的当前真实内容
- 处理测试启动路径与 `durable_resume` 依赖问题，恢复完整一键验证

## 更新时间

- 2026-04-15

## 更新人

- Codex（基于仓库审计）

## 建议更新方式

每次更新时，优先补充以下信息：

- 当前阶段
- 本阶段目标
- 已完成事项
- 进行中事项
- 阻塞项
- 下一步动作
- 更新时间
- 更新人
