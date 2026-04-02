# 二人周常活动

## Overview

A 7-day recurring weekly activity for the 2-player fishing mode where players complete daily tasks to earn activity points and unlock milestone rewards. The activity is structured into seasons (S1-S7) with varying task requirements, fish values, and reward distributions to maintain economic balance.

## Terminology

- `activity_points`: 活跃值 (activity points; 活跃度)
  Points earned by completing daily tasks, used to progress through weekly milestone rewards.
- `cannon_multiplier`: 炮倍 (cannon multiplier)
  The multiplier applied to the base cost of a shot, often used as a requirement for task completion (e.g., catch fish with 10k multiplier).
- `weekly_tasks`: 周常任务 (weekly tasks)
  A set of tasks refreshed weekly, divided into 7 days of unlocking content.

## Entities

- `daily_task`: 每日任务
  Individual objectives such as logging in, consuming gold, or catching specific fish types.
- `milestone_reward`: 活跃奖励
  Rewards granted when the total weekly activity points reach specific thresholds (e.g., 100, 400, 1000).
- `freeze_card`: 冰冻卡
  A consumable item (ID 1170) often given as a reward for individual tasks.
- `summon_card`: 召唤卡
  A consumable item (ID 1169) often given as a reward for individual tasks.

## Rules

1. 任务解锁规则
   Tasks are categorized by day (Day 1 to Day 7). Tasks for future days are locked and cannot be completed or progressed in advance.
   Condition: Current day < Task day
2. 未解锁天数交互
   Clicking on a locked day will only display a text prompt; no other interaction is permitted.
   Condition: Day is locked
3. 分期设定
   The value of weekly tasks and special fish must be configured differently for each season (S1 through S7).
4. 流水转化活跃度
   Player turnover (spending) is converted into activity progress based on expected turnover task damage reduction and activity value.
   Condition: Spending gold in 2P fishing mode

## Config Overview

- Task IDs and descriptions per day (Day 1-7)
- Activity point values per task
- Milestone reward thresholds (e.g., 100, 200, 400, 600, 700, 800, 1000)
- Item IDs for rewards (e.g., 1170, 1169, 1452)
- Seasonal configuration for S1-S7

## Open Questions / Assumptions

### Open Questions

- Will the red dot notification and the lock icon appear simultaneously on locked days?
- What is the specific text prompt displayed when clicking a locked day?

### Assumptions

- The activity resets every 7 days.
- The 'Paid Activity' mentioned in sheets refers to tasks requiring gold consumption or high-tier gameplay.
- The 'S' prefix in tracking refers to Seasons or Periods.
