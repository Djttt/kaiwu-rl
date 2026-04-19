#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum DQN agent.
清扫大作战 DQN 配置。
"""


class Config:

    # Feature dimensions (Sync with PPO for consistency)
    # 特征维度（与 PPO 保持同步）
    LOCAL_VIEW_LEN = 21 * 21
    LEGAL_ACTION_LEN = 8

    # Global state sub-features
    # 全局状态子特征
    BASE_GLOBAL_STATE_LEN = 12

    CHARGER_FEATURE_LEN = 4

    CHARGER_PRIMARY_KEYS = (
        "chargers",
        "charger_list",
        "charger_positions",
        "charger_pos",
    )

    NPC_PRIMARY_KEYS = (
        "official_robots",
        "robots",
        "robot_list",
        "npcs",
        "npc_list",
        "enemy_robots",
        "opponent_robots",
    )

    NPC_DIR_BINS = 8
    NPC_VEL_DIR_BINS = 8
    NPC_FEATURE_LEN = 5 + NPC_DIR_BINS + NPC_VEL_DIR_BINS

    TRAJECTORY_FEATURE_LEN = 4

    DIRT_MAP_SIZE = 16
    DIRT_MAP_FEATURE_LEN = DIRT_MAP_SIZE * DIRT_MAP_SIZE

    MAP_MEMORY_SIZE = 16
    MAP_MEMORY_CHANNELS = 2
    MAP_MEMORY_FEATURE_LEN = MAP_MEMORY_SIZE * MAP_MEMORY_SIZE * MAP_MEMORY_CHANNELS

    BFS_RECOMPUTE_INTERVAL = 3

    GLOBAL_STATE_LEN = (
        BASE_GLOBAL_STATE_LEN
        + CHARGER_FEATURE_LEN
        + NPC_FEATURE_LEN
        + TRAJECTORY_FEATURE_LEN
        + DIRT_MAP_FEATURE_LEN
        + MAP_MEMORY_FEATURE_LEN
    )

    FEATURES = [LOCAL_VIEW_LEN, GLOBAL_STATE_LEN, LEGAL_ACTION_LEN]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    LOCAL_VIEW_SHAPE = (1, 21, 21)

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value (DQN computes Q-values, effectively 1 output head)
    # 单头价值（DQN 计算 Q 值，即一个输出头）
    VALUE_NUM = 1

    # DQN hyperparameters
    # DQN 超参数
    GAMMA = 0.99
    BATCH_SIZE = 64
    REPLAY_BUFFER_CAPACITY = 500000
    TARGET_UPDATE_FREQ = 1000
    LEARNING_RATE = 0.0001
    
    # Epsilon greedy parameters
    # Epsilon 贪心参数
    EPSILON_START = 1.0
    EPSILON_END = 0.15
    EPSILON_DECAY = 250000
    
    # Soft update tau
    # 目标网络软更新系数
    TAU = 0.005

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # Hard unstuck switch
    # 硬脱困开关
    ENABLE_HARD_UNSTUCK = True
    HARD_UNSTUCK_TRIGGER_STEPS = 4
    HARD_UNSTUCK_BAN_STEPS = 2
