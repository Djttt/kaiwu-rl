---
name: sweep-battle
description: 清扫大作战项目知识库。当用户询问关于该强化学习竞赛项目的任何问题时触发，包括：环境配置、观测空间、动作空间、奖励设计、特征工程、模型结构、PPO算法实现、充电策略、NPC躲避、地图配置、评估任务、代码优化等。只要用户提到"清扫"、"小悟"、"充电桩"、"官方机器人"、"PPO"、"特征处理"、"奖励函数"、"开悟平台"等关键词，务必触发此 Skill。
---

# 清扫大作战 —— 项目知识库

## 任务概述

**目标**：操控"小悟"机器人，在规定步数和电量耗尽前，清扫尽可能多的地面污渍，最大化清扫得分。

| 要素 | 说明 |
|------|------|
| 地图大小 | 128×128 栅格，左上角为原点 (0,0)，x 向右，z 向下 |
| 视野范围 | 以智能体为中心的 21×21 区域（各方向延伸 10 格） |
| 初始电量 | 200（可配置，范围 100～999） |
| 每步耗电 | 1 格电量 |
| 得分规则 | 清扫地面数量 = 任务得分（区别于 RL 奖励 reward） |
| 任务终止 | 撞上官方机器人 / 电量耗尽 / 达到最大步数 |

---

## 环境配置（train_env_conf.toml）

```toml
[env_conf]
map = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 训练地图列表
map_random = false        # true=随机抽取, false=顺序轮换
robot_count = 4           # 官方机器人数量（1～4）
charger_count = 4         # 充电桩数量（1～4）
max_step = 1000           # 最大步数（1～2000）
battery_max = 200         # 满电电量（100～999）
```

- **共 15 张地图**：10 张开放供训练/评估（编号 1～10），5 张隐藏用于最终测评
- 配置错误会导致训练任务变为"失败"，查看 env 模块错误日志排查

---

## 观测空间（Observation）

### 数据结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `step_no` | int32 | 当前步数 |
| `frame_state` | FrameState | 帧状态数据（含智能体坐标、电量等） |
| `env_info` | EnvInfo | 环境信息（含充电桩全局坐标、官方机器人位置） |
| `map_info` | list[list[int]] | 21×21 局部地图：0=障碍物, 1=已清扫, 2=污渍 |
| `legal_act` | list[int] | 合法动作掩码，长度 8，通常全为 1 |

> 充电桩提供**全局绝对位置**信息，官方机器人位置在视野内可见。

### 接口调用

```python
# 重置环境
env_obs = env.reset(usr_conf=usr_conf)

# 执行动作
env_reward, env_obs = env.step(hero_actions)

# env_reward 包含：frame_no, env_id, reward（当前得分）
# env_obs 包含：env_id, frame_no, observation, extra_info, terminated, truncated
```

---

## 动作空间

8 个离散移动方向（0～7）：

| 动作值 | 方向 | 向量 (dx, dy) |
|--------|------|--------------|
| 0 | 右 (→) | (1, 0) |
| 1 | 右上 (↗) | (1, -1) |
| 2 | 上 (↑) | (0, -1) |
| 3 | 左上 (↖) | (-1, -1) |
| 4 | 左 (←) | (-1, 0) |
| 5 | 左下 (↙) | (-1, 1) |
| 6 | 下 (↓) | (0, 1) |
| 7 | 右下 (↘) | (1, 1) |

**移动规则**：
1. 直线移动：目标格可通行即可移动
2. 斜向防穿角：目标格可通行，且水平/垂直方向至少一条边可通行
3. 碰撞处理：无法移动时原地停留，但仍消耗 1 步 + 1 电量

---

## 基线智能体实现

### 特征向量（69 维）

| 部分 | 维度 | 含义 |
|------|------|------|
| `local_view` | 49 | 7×7 局部地图（归一化到 [0,1]），0=障碍, 1=已清扫, 2=污渍 |
| `global_state` | 12 | 步数归一化、电量比、清扫进度、剩余污渍比、坐标归一化、4方向污渍距离、最近污渍距离、是否在接近污渍 |
| `legal_action` | 8 | 8方向合法动作掩码 |

```python
def observation_process(self, env_obs):
    feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
    obs_data = ObsData(
        feature=list(feature),
        legal_action=legal_action,
    )
    remain_info = {"reward": reward}
    return obs_data, remain_info
```

### 基线奖励设计

| 奖励项 | 含义 | 数值 |
|--------|------|------|
| `cleaning_reward` | 每步清扫格子数奖励 | +0.1 × 清扫格数 |
| `step_penalty` | 每步时间惩罚（鼓励效率） | -0.001 |

```
reward = cleaning_reward + step_penalty
```

时序处理：使用 GAE（Config.GAMMA, Config.LAMDA）计算 advantage 与 reward_sum。

---

## 算法：PPO（Proximal Policy Optimization）

### 网络结构（Actor-Critic）

```
输入 (69D) → MLP骨干 (69→128→64) → Actor头 (64→8)  → 动作 logits
                                   → Critic头 (64→1) → 状态价值
```

- **合法动作掩码**：对非法动作 logits 加大负数，再 softmax
- **训练采样**：multinomial 随机采样
- **评估推断**：argmax 最大概率

### 训练流程

1. **交互采样**：Agent 与环境交互生成 SampleData（obs, action, prob, value, reward）
2. **后处理**：填充 next_value，用 GAE 计算 advantage 与 reward_sum
3. **PPO 更新**：策略损失 + 价值损失 + 熵正则
4. **保存/评估**：每 30 分钟保存一个模型，在验证地图上评估性能

### 损失函数

```
total_loss = vf_coef × value_loss + policy_loss - beta × entropy_loss
```

| 损失项 | 说明 |
|--------|------|
| `value_loss` | Clipped 价值函数损失 |
| `policy_loss` | PPO Clipped surrogate 目标 |
| `entropy_loss` | 动作熵正则化（鼓励探索） |

### 监控指标（monitor_builder.py）

| 指标 | 说明 |
|------|------|
| `total_loss` | 总损失 |
| `policy_loss` | 策略损失 |
| `value_loss` | 价值损失 |
| `entropy_loss` | 熵损失 |
| `reward` | 累计回报 |

---

## 优化方向指南

### 1. 特征工程扩展

| 方向 | 具体内容 |
|------|----------|
| **充电桩特征** | 充电桩全局坐标、欧氏距离、BFS最短路径距离 |
| **NPC特征** | 官方机器人位置、速度方向、危险接近方向、距离 |
| **轨迹特征** | 历史移动轨迹，避免重复路径 |
| **全局污渍图** | 粗粒度网格表示全局污渍分布（如 16×16 下采样） |
| **地图记忆** | 记录已探索区域和未清扫区域的全局 bitmap |

### 2. 奖励函数设计

| 方向 | 具体内容 |
|------|----------|
| **充电桩导航** | 低电量时奖励接近充电桩，到达时给予大额奖励 |
| **NPC 躲避** | 与官方机器人距离过近时给予惩罚 |
| **探索奖励** | 奖励进入未探索区域（访问计数 / RND 内在奖励） |
| **效率奖励** | 奖励连续清扫行为，惩罚空闲踩空 |
| **多头奖励** | 分解清扫 / 电量 / 安全等多目标，分别 scale |

### 3. 模型结构升级

| 方向 | 具体内容 |
|------|----------|
| **CNN 分支** | 对 21×21 局部地图用 CNN 处理，保留空间结构信息 |
| **注意力机制** | 动态关注充电桩、NPC、污渍等实体的 cross-attention |
| **多头价值** | 分解清扫 / 充电 / 安全的价值估计，减少价值函数方差 |
| **RND 网络** | 随机网络蒸馏，添加好奇心驱动促进探索 |
| **独立 Critic** | 策略网络与价值网络分离，减少梯度干扰 |
| **LSTM/GRU** | 处理部分可观测问题，记忆历史信息 |

### 4. 行为策略

| 方向 | 具体内容 |
|------|----------|
| **充电策略** | 根据剩余电量和到充电桩距离动态决策，预留返程电量 |
| **躲避策略** | 根据 NPC 位置/速度预测危险区域，规划安全路径 |
| **清扫规划** | 螺旋覆盖、分区清扫，减少重复踩踏空格 |
| **泛化训练** | 全 10 张地图 + `map_random = true`，提升跨地图泛化 |

---

## 环境监控指标

| 中文名 | 英文名 | 指标名 | 说明 |
|--------|--------|--------|------|
| 得分 | score | `total_score` | 任务结束时总积分 |
| 得分 | score | `clean_score` | 清扫得分 |
| 步数 | steps | `max_step` | 最大步数配置 |
| 步数 | steps | `finished_steps` | 实际使用步数 |
| 充电 | charge | `remaining_charge` | 任务结束剩余电量 |
| 充电 | charge | `charge_count` | 每局充电次数 |
| 充电 | charge | `total_charger` | 充电桩总数 |
| 地图 | map | `total_map` | 地图总数 |
| 地图 | map | `map_random` | 是否随机地图 |

---

## 评估任务配置

```ini
[env_conf]
robot_count: 4    # 官方机器人数量（1～4）
charger_count: 4  # 充电桩数量（1～4）
max_step: 1000    # 最大步数（1～2000）
battery_max: 200  # 最大电量（100～999）
```

| 任务状态 | 说明 |
|----------|------|
| 已完成 | 撞到官方机器人 / 电量耗尽 / 达到最大步数 |
| 异常 | 各种原因导致的异常终止 |

**泛化性建议**：
- 训练时使用全部 10 张地图（`map_random = true`）
- 评估时在多张不同地图测试，确保跨地图适应性
- 最终测评在 5 张隐藏地图上进行，务必避免过拟合

---

## 关键注意事项

1. **得分 ≠ 奖励**：`env_reward.reward` 是清扫格数（评估指标），RL 的 `reward` 是人工设计的奖励函数，两者分开处理
2. **电量管理**：每步强制消耗 1 电量（包括碰墙原地不动），电量归零即任务终止
3. **斜向移动**：需要检查防穿角条件，否则会卡在角落原地空耗电量
4. **模型保存**：平台限制频率，默认每 30 分钟保存一次，不要过度频繁保存
5. **DIY 算法**：可在 `diy` 模板文件夹中完全自定义算法，不限于 PPO

---

## 开发框架详解

### 环境接口（Environment API）

#### `env.reset(usr_conf)`

重置环境，返回初始观测。

```python
obs, state = env.reset(usr_conf=usr_conf)
# obs: dict，环境观测信息
# state: dict，环境全局信息（EnvInfo）
```

#### `env.step(act, stop_game=False)`

执行动作，完成状态转移。

```python
frame_no, _obs, score, terminated, truncated, _state = env.step(act)
```

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `frame_no` | int | 当前帧号 |
| `_obs` | dict | 当前帧观测信息 |
| `score` | int | 当前帧得分（清扫格数） |
| `terminated` | bool | 是否终止（碰撞/电量耗尽） |
| `truncated` | bool | 是否截断（达到最大步数/异常） |
| `_state` | dict | 当前帧全局状态信息 |

---

### 数据结构定义（definition.py）

开发目录：`<智能体文件夹>/feature/definition.py`

使用 `create_cls` 动态定义三类核心数据结构：

```python
# 观测数据：agent.predict() 的输入
ObsData = create_cls("ObsData",
    feature=None,
    legal_action=None,
)

# 动作数据：agent.predict() 的输出
ActData = create_cls("ActData",
    action=None,
    prob=None,
)

# 训练样本：agent.learn() 的输入
SampleData = create_cls("SampleData",
    npdata=None
)
```

#### 样本序列化（分布式训练必须实现）

```python
# SampleData → Numpy（用于网络传输），需加 @attached 装饰器
@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata

# Numpy → SampleData（接收端还原），需加 @attached 装饰器
@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
```

> ⚠️ 这两个函数互为反函数，**必须加 `@attached` 装饰器**，否则分布式框架无法调用。

---

### 特征处理（agent.py）

#### `observation_process`：原始观测 → ObsData

```python
def observation_process(self, obs, state=None):
    # obs: Observation 类型（env.reset/step 返回）
    # state: EnvInfo 类型（可选）
    # 建议将大量特征处理逻辑封装在 preprocessor.py 中
    feature, legal_action, reward = self.preprocessor.feature_process(obs, self.last_action)
    return ObsData(feature=list(feature), legal_action=legal_action)
```

#### `action_process`：ActData → 环境动作

```python
def action_process(self, act_data):
    # 将智能体输出转换为 env.step() 可接受的格式
    return act_data.action
```

---

### 奖励设计（definition.py）

```python
def reward_shaping(obs, _obs, state, _state):
    # 参数不限，可使用任意环境信息和先验知识
    # 返回 float 类型的奖励值
    reward = ...
    return reward
```

> Score（env 返回的清扫格数）≠ Reward（此处人工设计），设计时注意区分。

---

### 样本处理（definition.py）

```python
@attached
def sample_process(self, list_game_data):
    # list_game_data: list[Frame]，一个 episode 的轨迹帧列表
    return [SampleData(**i.__dict__) for i in list_game_data]
```

---

### 智能体核心接口（agent.py）

#### `predict`：训练时推断（随机采样）

```python
@predict_wrapper
def predict(self, list_obs_data, list_state=None):
    # 输入：list[ObsData]
    # 输出：list[ActData]
    return [ActData(action=..., prob=...)]
```

#### `exploit`：评估时推断（贪心选最优）

```python
@exploit_wrapper
def exploit(self, observation):
    # 输入：dict，原始观测
    # 输出：list，环境可直接使用的动作列表
    return action
```

#### `learn`：模型训练

```python
def learn(self, list_sample_data):
    # 单机训练：手动在 workflow 中调用，直接训练模型
    # 分布式训练：框架自动循环调用（同时也用于发送样本到样本池）
    self.algo.learn(list_sample_data)
```

#### `load_model` / `save_model`

```python
@load_model_wrapper
def load_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))

@save_model_wrapper
def save_model(self, path=None, id="1"):
    model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
    model_state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
    torch.save(model_state_dict_cpu, model_file_path)
```

> 文件名中必须包含 `model.ckpt-{id}` 字段；保存和加载的文件名必须一致。

---

### 模型开发（model/model.py）

继承 `torch.nn.Module`，符合 PyTorch 规范：

```python
class Model(nn.Module):
    def __init__(self, state_shape, action_shape=0, softmax=False):
        super().__init__()
        # 自定义网络层
```

---

### 算法开发（algorithm/algorithm.py）

```python
def learn(self, list_sample_data):
    # list_sample_data: list[SampleData]
    loss = ...           # 基于算法计算 loss
    loss.backward()      # 反向传播
    self.optimizer.step()  # 梯度更新
```

---

### 训练工作流（workflow.py）

```python
@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]

    for epoch in range(epoch_num):
        for g_data in run_episodes(episode_num, env, agent, logger, monitor):
            agent.learn(g_data)   # 训练（或分布式发送样本）
            g_data.clear()        # 清空，确保下轮样本是新的

        # 按时间间隔保存模型（默认 300s = 5min，平台限制建议 30min）
        if now - last_save_model_time >= 300:
            agent.save_model()
            last_save_model_time = now
```

#### 单局 episode 核心循环

```python
while not done:
    act_data = agent.predict([obs_data])[0]
    act = agent.action_process(act_data)
    frame_no, _obs, score, terminated, truncated, _state = env.step(act)

    _obs_data = agent.observation_process(_obs, _state)
    reward = reward_shaping(obs_data, _obs_data, state, _state)
    done = terminated or truncated

    frame = Frame(obs=obs_data.feature, _obs=_obs_data.feature,
                  act=act, rew=reward, done=done)
    collector.append(frame)

    if done:
        yield sample_process(collector)  # 返回样本给 agent.learn()
        break

    obs_data, obs, state = _obs_data, _obs, _state
```

---

### 文件目录结构

```
<智能体文件夹>/
├── agent.py                    # 智能体：predict/exploit/learn/load_model/save_model
├── feature/
│   ├── definition.py           # 数据结构、奖励设计、样本处理
│   └── preprocessor.py         # 特征工程（observation → feature vector）
├── model/
│   └── model.py                # 神经网络模型（继承 nn.Module）
├── algorithm/
│   └── algorithm.py            # 强化学习算法（learn 函数）
├── workflow.py                 # 训练工作流（workflow 函数）
└── diy/                        # 自定义算法模板（可选）
```
