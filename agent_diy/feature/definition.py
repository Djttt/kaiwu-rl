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

    在消融实验的基础上，加回了“NPC 生存约束”与“电量保持约束”：
    让机器人能在高效清扫的同时，学会保命和回家。
    """
    cur = _split_feature(obs)
    nxt = _split_feature(_obs)

    # 1) 清扫奖励 (主线任务)
    score_delta = kwargs.get("score_delta", 0.0)
    step_score = np.clip(_to_float(score_delta, 0.0), 0.0, 10.0)
    cleaning_reward = 0.1 * step_score

    # 2) 时间惩罚 (鼓励高效)
    step_penalty = -0.002

    # 3) 生存约束：躲避 NPC
    # 距离越近、NPC朝向自己速度越快，惩罚越大
    npc_dist = _to_float(nxt["npc"][2])
    npc_approach = _to_float(nxt["npc"][3])
    npc_danger = _to_float(nxt["npc"][4])
    npc_penalty = -0.05 * npc_danger * (1.0 - npc_dist) * (0.5 + npc_approach)

    # 4) 资源约束：电量管理与返航
    battery_prev = _to_float(cur["base"][1])
    battery_ratio = _to_float(nxt["base"][1])
    charger_next = _to_float(nxt["charger"][3])
    
    # 风险惩罚：当预期返航距离（+0.05的冗余）超过当前电量比例时，施加惩罚，迫使模型在电量不足时必须往充电桩走
    return_risk = max(0.0, charger_next + 0.05 - battery_ratio)
    battery_risk_penalty = -0.3 * return_risk

    # 充电奖励：成功充上电的瞬间给出显著正反馈，鼓励充电行为
    charge_gain = max(0.0, battery_ratio - battery_prev)
    charge_bonus = 0.15 * charge_gain

    # 总奖励
    reward = cleaning_reward + step_penalty + npc_penalty + battery_risk_penalty + charge_bonus

    if not np.isfinite(reward):
        return 0.0

    return float(np.clip(reward, -1.0, 1.0))
