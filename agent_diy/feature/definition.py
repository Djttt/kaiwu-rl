#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and helper functions for Robot Vacuum DQN.
清扫大作战数据类定义。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config


# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, action probabilities (or Q-values in DQN)
# 动作数据：action 为采样动作，prob 为动作概率/Q值，value 为状态价值（可选）
ActData = create_cls(
    "ActData",
    action=None,
    prob=None,
    value=None,
)

# SampleData: stores transitions (s, a, r, s', done, masks)
# 训练样本数据：存储 DQN 经验回放所需的一条转移。
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    next_obs=Config.DIM_OF_OBSERVATION,
    next_legal_action=Config.ACTION_NUM,
    done=1,
)


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
    """
    cur = _split_feature(obs)
    nxt = _split_feature(_obs)

    # 1) Progress / Cleaning Term
    clean_progress_gain = _to_float(nxt["base"][2] - cur["base"][2])
    progress_term = 4.0 * clean_progress_gain

    # 2) Score Term
    score_delta = kwargs.get("score_delta", 0.0)
    step_score = np.clip(_to_float(score_delta, 0.0), 0.0, 10.0)
    score_term = 0.08 * step_score

    # 3) Dirt-density reduction
    dirt_drop = _to_float(np.mean(cur["dirt_map"]) - np.mean(nxt["dirt_map"]))
    dirt_term = 0.6 * dirt_drop

    # 4) Exploration bonus
    explored_gain = max(
        0.0,
        _to_float(np.mean(nxt["map_memory_explored"]) - np.mean(cur["map_memory_explored"])),
    )
    exploration_term = 0.2 * explored_gain

    # 5) Battery-aware charger guidance
    battery_ratio = _to_float(nxt["base"][1])
    low_battery = max(0.0, 0.55 - battery_ratio) / 0.55
    charger_prev = _to_float(cur["charger"][3])
    charger_next = _to_float(nxt["charger"][3])
    charger_term = 0.18 * low_battery * (charger_prev - charger_next)

    # 6) Energy risk penalty
    return_risk = max(0.0, charger_next + 0.08 - battery_ratio)
    battery_risk_penalty = -0.25 * return_risk * (1.0 + low_battery)

    # 7) NPC risk penalty
    npc_dist = _to_float(nxt["npc"][2])
    npc_approach = _to_float(nxt["npc"][3])
    npc_danger = _to_float(nxt["npc"][4])
    npc_penalty = -0.03 * npc_danger * (1.0 - npc_dist) * (0.5 + npc_approach)

    # 8) Trajectory shaping
    revisit_ratio = _to_float(nxt["traj"][0])
    backtrack_flag = _to_float(nxt["traj"][1])
    turn_rate = _to_float(nxt["traj"][2])
    progress_eff = _to_float(nxt["traj"][3])
    traj_term = -0.012 * revisit_ratio - 0.012 * backtrack_flag - 0.006 * turn_rate + 0.012 * progress_eff

    # Survival and step penalty
    survival_term = 0.002
    step_penalty = -0.0015

    reward = (
        score_term
        + progress_term
        + dirt_term
        + exploration_term
        + charger_term
        + battery_risk_penalty
        + npc_penalty
        + traj_term
        + survival_term
        + step_penalty
    )

    if not np.isfinite(reward):
        return 0.0

    return float(np.clip(reward, -1.0, 1.0))
