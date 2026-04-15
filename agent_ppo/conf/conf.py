#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
清扫大作战 PPO 配置。
"""


class Config:

    # Feature dimensions
    # 特征维度
    LOCAL_VIEW_LEN = 7 * 7
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

    LOCAL_VIEW_SHAPE = (1, 7, 7)

    # Action space: 8 directional moves
    # 动作空间：8个方向移动
    ACTION_NUM = 8

    # Single-head value
    # 单头价值
    VALUE_NUM = 1

    # PPO hyperparameters
    # PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.001
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5
