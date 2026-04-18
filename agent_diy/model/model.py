#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY DQN Model (Dueling DQN with CNN encoder).
清扫大作战 DIY DQN 模型实现（基于 CNN 的 Dueling DQN）。
"""

import torch
import torch.nn as nn
from agent_diy.conf.conf import Config


class Model(nn.Module):
    """DQN Model with CNN encoder for local map and MLP for global state.

    DQN 模型：使用 CNN 编码局部地图，使用 MLP 编码全局状态。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum_dqn"
        self.device = device

        # CNN for local view (21x21 input)
        # 局部视野卷积编码 (21x21 输入)
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Flat dimension: 32 channels * 21 * 21
        flat_dim = 32 * Config.LOCAL_VIEW_SHAPE[1] * Config.LOCAL_VIEW_SHAPE[2]

        # MLP for global state
        # 全局状态 MLP 编码
        self.global_mlp = nn.Sequential(
            nn.Linear(Config.GLOBAL_STATE_LEN, 256),
            nn.ReLU(),
        )

        # Dueling DQN architecture
        # Dueling DQN 结构：优势流与价值流
        common_dim = flat_dim + 256

        self.advantage_stream = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Linear(256, Config.ACTION_NUM),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(common_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=1.414)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_obs(self, s):
        x = s.to(torch.float32)
        local_view = x[:, : Config.LOCAL_VIEW_LEN].view(-1, 1, 21, 21)
        global_state = x[:, Config.LOCAL_VIEW_LEN : Config.LOCAL_VIEW_LEN + Config.GLOBAL_STATE_LEN]
        return local_view, global_state

    def forward(self, s, inference=False):
        """Forward pass to compute Q-values.

        前向传播计算各动作的 Q 值。
        """
        local_view, global_state = self._split_obs(s)

        l_feat = self.local_cnn(local_view)
        g_feat = self.global_mlp(global_state)
        combined = torch.cat([l_feat, g_feat], dim=1)

        advantage = self.advantage_stream(combined)
        value = self.value_stream(combined)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
