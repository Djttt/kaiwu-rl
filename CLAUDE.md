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
env_obs → Preprocessor.feature_process() → 866D feature + legal_action
                                                 ↓
                                        Agent.observation_process() → ObsData
                                                 ↓
                                        Agent.predict() / exploit() → ActData
                                                 ↓
                                        Agent.learn() → Algorithm.learn() → PPO update
```

**模型结构**（model.py）：

- 观测 866D = 局部地图 49D + 全局状态 809D + 合法动作掩码 8D
- `_split_obs()` 将 866D 拆分为 `local_view`（49D → reshape 成 1×7×7）和 `global_state`（809D）
- `_Encoder`：局部地图走 CNN（1→16→32 → 1568→128），全局状态走 MLP（809→256→128），拼接成 256D
- **Actor 和 Critic 各自有独立 Encoder**，不共享 backbone，减少梯度干扰
- Actor head：`[B,256] → 128 → 8`（logits）
- Critic head：`[B,256] → 128 → 1`（value）

```
输入 s [B, 866]
  │
  ├── local_view [B, 49] → reshape [B, 1, 7, 7]
  │     CNN: Conv2d(1→16→32) → Flatten → Linear(1568→128)
  │
  ├── global_state [B, 809]
  │     MLP: Linear(809→256→128)
  │
  └── concat → [B, 256]
        │
        ├──→ Actor_encoder → Actor_head(256→128→8) → [B, 8] logits
        └──→ Critic_encoder → Critic_head(256→128→1) → [B, 1] value
```

**配置**（conf.py）：观测维度、PPO 超参数（GAMMA=0.99, LAMDA=0.95, CLIP_PARAM=0.2）

**特征修改说明（preprocessor.py）**：

- 总维度：866D = 49D 局部地图 + 809D 全局状态 + 8D 动作掩码
- 局部地图（49D）：21×21 视野中心裁剪 7×7
- 全局状态（809D）：
  - 基础状态（12D）：步数、电量、清扫进度、坐标、污渍距离等
  - 充电桩（4D）：最近充电桩坐标 + 欧氏距离 + BFS 最短路
  - NPC（21D）：相对位置/距离 + 接近风险方向 + 速度方向
  - 轨迹（4D）：回访率、折返、转向率、推进效率
  - 全局污渍图（256D）：16×16 粗粒度污渍分布
  - 地图记忆 bitmap（512D）：16×16 explored + 16×16 dirty 双通道
- 合法动作掩码（8D）：保持在模型外处理
- 稳健性：实体坐标采用“白名单 key 优先，关键词兜底”解析
- 性能优化：BFS 路径距离增加缓存与低频重算（默认每 3 步重算）

**奖励**（preprocessor.py）：`reward = 0.1 × 清扫格数 - 0.001`（清扫奖励 + 时间惩罚）

### agent_diy/ — 可自定义算法框架模板

与 agent_ppo 结构一致，但可完全自定义算法。

### conf/ — 配置文件

- `algo_conf_robot_vacuum.toml`：算法配置
- `app_conf_robot_vacuum.toml`：应用配置
- `configure_app.toml`：主配置

## 关键设计

1. **合法动作掩码**在模型外处理（`Agent._legal_soft_max`），模型只输入前 858D（49D 局部地图 + 809D 全局状态）
2. **Actor/Critic 分离**：避免共享 backbone 带来的梯度干扰
3. **局部地图裁剪**：21×21 视野中心裁剪 7×7，减少计算量
4. **PPO 损失**：`total_loss = 0.5 * value_loss + policy_loss - 0.001 * entropy_loss`

## 修改记录

| 日期       | 修改内容 |
|------------|----------|
| 2026-04-15 | 初始提交：项目框架、PPO 基线实现 |
| 2026-04-15 | 模型重构：分离 Actor/Critic 编码器 + CNN 局部地图编码器 |
| 2026-04-15 | 特征工程升级：69D → 866D |
| 2026-04-15 | 创建 CLAUDE.md 记录代码架构 |
