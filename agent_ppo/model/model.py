#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Separate actor/critic network with CNN map encoder for Robot Vacuum.
清扫大作战独立 Actor-Critic + CNN 局部地图编码网络。
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    """Create a linear layer with orthogonal initialization.

    创建正交初始化的线性层。
    """
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=1.41421):
    """Create a convolution layer with orthogonal initialization.

    创建正交初始化的卷积层。
    """
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class _Encoder(nn.Module):
    """Encode local map and global state with separate parameters.

    使用独立参数编码局部地图与全局状态。
    """

    def __init__(self):
        super().__init__()
        self.local_cnn = nn.Sequential(
            _make_conv(1, 16, gain=1.0),
            nn.ReLU(),
            _make_conv(16, 32, gain=1.0),
            nn.ReLU(),
            nn.Flatten(),
            _make_fc(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.global_mlp = nn.Sequential(
            _make_fc(Config.GLOBAL_STATE_LEN, 32),
            nn.ReLU(),
            _make_fc(32, 32),
            nn.ReLU(),
        )

    def forward(self, local_view, global_state):
        local_feat = self.local_cnn(local_view)
        global_feat = self.global_mlp(global_state)
        return torch.cat([local_feat, global_feat], dim=1)


class Model(nn.Module):
    """Independent actor-critic model for Robot Vacuum.

    清扫大作战独立 Actor-Critic 模型。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        act_num = Config.ACTION_NUM  # 8

        # Separate actor and critic encoders to reduce gradient interference.
        # 分离 Actor 和 Critic 编码器，减少梯度干扰。
        self.actor_encoder = _Encoder()
        self.critic_encoder = _Encoder()

        self.actor_head = nn.Sequential(
            _make_fc(96, 64),
            nn.ReLU(),
            _make_fc(64, act_num, gain=0.01),
        )

        self.critic_head = nn.Sequential(
            _make_fc(96, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=0.01),
        )

    def _split_obs(self, s):
        x = s.to(torch.float32)
        local_view = x[:, : Config.LOCAL_VIEW_LEN].view(-1, *Config.LOCAL_VIEW_SHAPE)
        global_state = x[:, Config.LOCAL_VIEW_LEN : Config.LOCAL_VIEW_LEN + Config.GLOBAL_STATE_LEN]
        return local_view, global_state

    def forward(self, s, inference=False):
        """Forward pass.

        前向传播。
        """
        local_view, global_state = self._split_obs(s)

        actor_feat = self.actor_encoder(local_view, global_state)
        critic_feat = self.critic_encoder(local_view, global_state)

        logits = self.actor_head(actor_feat)
        value = self.critic_head(critic_feat)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
