#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation for Robot Vacuum.
清扫大作战数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, greedy action, action probabilities, state value
# 动作数据：action 为采样动作，d_action 为贪心动作，prob 为动作概率，value 为状态价值
ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
)

# SampleData: int values are treated as dimensions by the framework
# 训练样本数据：字段值为 int 时框架自动按维度处理
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,  # feature vector / 特征向量
    legal_action=Config.ACTION_NUM,  # 8D legal action mask / 合法动作掩码
    act=1,  # action index / 执行的动作
    reward=Config.VALUE_NUM,  # 1D reward / 奖励
    reward_sum=Config.VALUE_NUM,  # GAE td-lambda return
    done=1,
    value=Config.VALUE_NUM,  # 1D value estimate / 价值估计
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,  # 1D GAE advantage / GAE 优势
    prob=Config.ACTION_NUM,  # 8D action probabilities / 动作概率
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    计算 GAE 并填充 next_value。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute advantage and cumulative return using GAE(λ).

    使用 GAE(λ) 计算优势函数与累积回报。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def _as_1d_feature(feature):
    """Convert input feature to float32 1D numpy array.

    将输入特征转为 float32 一维数组。
    """
    arr = np.asarray(feature, dtype=np.float32).reshape(-1)
    if arr.size != Config.DIM_OF_OBSERVATION:
        raise ValueError(
            f"feature size mismatch: got {arr.size}, expect {Config.DIM_OF_OBSERVATION}"
        )
    return arr


def _split_feature(feature):
    """Split full feature into semantic blocks.

    将完整特征拆分为语义分块。
    """
    feat = _as_1d_feature(feature)

    local_end = Config.LOCAL_VIEW_LEN
    global_end = local_end + Config.GLOBAL_STATE_LEN

    global_state = feat[local_end:global_end]

    base_end = Config.BASE_GLOBAL_STATE_LEN
    charger_end = base_end + Config.CHARGER_FEATURE_LEN
    npc_end = charger_end + Config.NPC_FEATURE_LEN
    traj_end = npc_end + Config.TRAJECTORY_FEATURE_LEN
    dirt_end = traj_end + Config.DIRT_MAP_FEATURE_LEN
    memory_end = dirt_end + Config.MAP_MEMORY_FEATURE_LEN

    map_memory = global_state[dirt_end:memory_end]
    half_memory = Config.MAP_MEMORY_FEATURE_LEN // 2

    return {
        "base": global_state[:base_end],
        "charger": global_state[base_end:charger_end],
        "npc": global_state[charger_end:npc_end],
        "traj": global_state[npc_end:traj_end],
        "dirt_map": global_state[traj_end:dirt_end],
        "map_memory_explored": map_memory[:half_memory],
        "map_memory_dirty": map_memory[half_memory:],
    }


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_env_score_step(state):
    """Extract score signal from env reward payload.

    从环境 reward 负载中提取得分信号。
    """
    if state is None:
        return 0.0

    if isinstance(state, dict):
        for key in ("step_score", "reward", "score", "clean_score", "total_score"):
            if key in state:
                return _to_float(state.get(key), 0.0)

    if hasattr(state, "reward"):
        return _to_float(getattr(state, "reward"), 0.0)

    if hasattr(state, "step_score"):
        return _to_float(getattr(state, "step_score"), 0.0)

    if hasattr(state, "score"):
        return _to_float(getattr(state, "score"), 0.0)

    return 0.0


def _extract_total_dirt(state):
    """Extract total dirt count from env info.

    从环境信息中提取全局污渍总数。
    """
    if isinstance(state, dict):
        return max(_to_float(state.get("total_dirt", 1.0), 1.0), 1.0)
    return 1.0


def reward_shaping(obs, _obs, state=None, _state=None, **kwargs):
    """Compute shaped RL reward from current/next feature states.

    依据当前/下一时刻特征计算强化学习 reward。

    Notes:
    - Score comes from environment and is only one component here.
      环境 score 仅作为 reward 的一个组成部分。
    - Reward integrates task heuristics to improve training signal.
      reward 融合开发者策略经验以增强训练信号。
    """
    cur = _split_feature(obs)
    nxt = _split_feature(_obs)

    # 2) Cleaning/progress component
    # 2) 清扫推进项
    clean_progress_gain = _to_float(nxt["base"][2] - cur["base"][2])
    progress_term = 4.0 * clean_progress_gain

    # 1) Incremental score component (delta score only)
    # 1) 增量得分项（只使用单步新增得分）
    score_delta = kwargs.get("score_delta")
    if score_delta is None:
        # Fallback: approximate delta from cleaning progress ratio.
        # 回退逻辑：用清扫进度增量近似单步得分。
        score_delta = clean_progress_gain * _extract_total_dirt(_state)

        # Backward compatibility: if caller provides explicit step_score, prefer it.
        # 向后兼容：若调用方提供 step_score，则优先使用。
        if score_delta <= 0.0 and isinstance(state, dict) and "step_score" in state:
            score_delta = _extract_env_score_step(state)

    step_score = np.clip(_to_float(score_delta, 0.0), 0.0, 10.0)
    score_term = 0.08 * step_score

    # 3) Dirt-density reduction on global map
    # 3) 全局污渍密度下降项
    dirt_drop = _to_float(np.mean(cur["dirt_map"]) - np.mean(nxt["dirt_map"]))
    dirt_term = 0.6 * dirt_drop

    # 4) Exploration bonus from map memory bitmap
    # 4) 地图记忆探索奖励
    explored_gain = max(
        0.0,
        _to_float(np.mean(nxt["map_memory_explored"]) - np.mean(cur["map_memory_explored"])),
    )
    exploration_term = 0.2 * explored_gain

    # 5) Battery-aware charger guidance
    # 5) 低电量充电引导
    battery_prev = _to_float(cur["base"][1])
    battery_ratio = _to_float(nxt["base"][1])
    low_battery = max(0.0, 0.55 - battery_ratio) / 0.55

    # Prefer BFS-distance signal for return-to-charger planning.
    # 优先使用 BFS 距离作为返航信号。
    charger_prev = _to_float(cur["charger"][3])
    charger_next = _to_float(nxt["charger"][3])
    charger_term = 0.18 * low_battery * (charger_prev - charger_next)

    # Penalize energy risk: estimated return distance exceeds battery margin.
    # 惩罚电量风险：估计返航距离超过电量余量。
    return_risk = max(0.0, charger_next + 0.08 - battery_ratio)
    battery_risk_penalty = -0.25 * return_risk * (1.0 + low_battery)

    # Reward successful charging events.
    # 奖励成功充电行为。
    charge_gain = max(0.0, battery_ratio - battery_prev)
    charge_bonus = 0.12 * charge_gain

    # 6) NPC risk penalty (distance + approach risk)
    # 6) NPC 风险惩罚（距离 + 接近趋势）
    npc_dist = _to_float(nxt["npc"][2])
    npc_approach = _to_float(nxt["npc"][3])
    npc_danger = _to_float(nxt["npc"][4])
    npc_penalty = -0.03 * npc_danger * (1.0 - npc_dist) * (0.5 + npc_approach)

    # 7) Trajectory shaping (avoid loops/backtracking, keep forward efficiency)
    # 7) 轨迹塑形（减少回环折返，提升推进效率）
    revisit_ratio = _to_float(nxt["traj"][0])
    backtrack_flag = _to_float(nxt["traj"][1])
    turn_rate = _to_float(nxt["traj"][2])
    progress_eff = _to_float(nxt["traj"][3])
    traj_term = -0.012 * revisit_ratio - 0.012 * backtrack_flag - 0.006 * turn_rate + 0.012 * progress_eff

    # Extra anti-stuck penalty when looping with very low progress efficiency.
    # 在低推进效率下反复回环时，增加防卡死惩罚。
    stuck_flag = 1.0 if (revisit_ratio > 0.75 and progress_eff < 0.15) else 0.0
    stuck_penalty = -0.03 * stuck_flag

    # 8) Survival-first small bonus + step efficiency penalty
    # 8) 生存优先的小奖励 + 时间惩罚
    survival_term = 0.002
    step_penalty = -0.0015

    reward = (
        score_term
        + progress_term
        + dirt_term
        + exploration_term
        + charger_term
        + battery_risk_penalty
        + charge_bonus
        + npc_penalty
        + traj_term
        + stuck_penalty
        + survival_term
        + step_penalty
    )

    if not np.isfinite(reward):
        return 0.0

    return float(np.clip(reward, -1.0, 1.0))
