# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**清扫大作战**强化学习竞赛 - 操控"小悟"机器人在 128×128 地图上清扫污渍。

- 地图：128×128 栅格，视野 21×21
- 动作：8 方向离散移动（0-7）
- 电量：200，每步-1
- 终止：撞机器人/电量耗尽/达最大步数

## 常用命令

```bash
# 运行训练/测试
python train_test.py
```

## 代码架构

### agent_ppo/ — PPO 算法实现（主力使用）

**数据流**：

```
env_obs → Preprocessor.feature_process() → 69D feature + legal_action
                                                 ↓
                                        Agent.observation_process() → ObsData
                                                 ↓
                                        Agent.predict() / exploit() → ActData
                                                 ↓
                                        Agent.learn() → Algorithm.learn() → PPO update
```

**模型结构**（model.py）：

- 观测 69D = 局部地图 49D + 全局状态 12D + 合法动作掩码 8D
- `_split_obs()` 将 69D 拆分为 `local_view`（49D → reshape 成 1×7×7）和 `global_state`（12D）
- `_Encoder`：局部地图走 CNN（1→16→32 → 1568→64），全局状态走 MLP（12→32→32），拼接成 96D
- **Actor 和 Critic 各自有独立 Encoder**，不共享 backbone，减少梯度干扰
- Actor head：`[B,96] → 64 → 8`（logits）
- Critic head：`[B,96] → 64 → 1`（value）

```
输入 s [B, 69]
  │
  ├── local_view [B, 49] → reshape [B, 1, 7, 7]
  │     CNN: Conv2d(1→16→32) → Flatten → Linear(1568→64)
  │
  ├── global_state [B, 12]
  │     MLP: Linear(12→32→32)
  │
  └── concat → [B, 96]
        │
        ├──→ Actor_encoder → Actor_head → [B, 8] logits
        └──→ Critic_encoder → Critic_head → [B, 1] value
```

**配置**（conf.py）：观测维度、PPO 超参数（GAMMA=0.99, LAMDA=0.95, CLIP_PARAM=0.2）

**特征**（preprocessor.py）：69D = 7×7 裁剪地图 + 12D 全局状态 + 8D 动作掩码

**奖励**（preprocessor.py）：`reward = 0.1 × 清扫格数 - 0.001`（清扫奖励 + 时间惩罚）

### agent_diy/ — 可自定义算法框架模板

与 agent_ppo 结构一致，但可完全自定义算法。

### conf/ — 配置文件

- `algo_conf_robot_vacuum.toml`：算法配置
- `app_conf_robot_vacuum.toml`：应用配置
- `configure_app.toml`：主配置

## 关键设计

1. **合法动作掩码**在模型外处理（`Agent._legal_soft_max`），模型只输入前 61D
2. **Actor/Critic 分离**：避免共享 backbone 带来的梯度干扰
3. **局部地图裁剪**：21×21 视野中心裁剪 7×7，减少计算量
4. **PPO 损失**：`total_loss = 0.5 * value_loss + policy_loss - 0.001 * entropy_loss`

## 修改记录

| 日期 | 修改内容 |
|------|----------|
| 2026-04-15 | 初始提交：项目框架、PPO 基线实现 |
| 2026-04-15 | 模型重构：分离 Actor/Critic 编码器 + CNN 局部地图编码器（model.py, conf.py） |
| 2026-04-15 | 创建 CLAUDE.md，记录代码架构和数据流 |