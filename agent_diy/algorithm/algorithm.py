#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY DQN Algorithm (Double DQN with Legal Action Masking).
清扫大作战 DIY DQN 算法实现（支持动作掩码的 Double DQN）。
"""

import torch
import torch.nn.functional as F
from agent_diy.conf.conf import Config


class Algorithm:
    """Double DQN Algorithm with legal action masking.

    Double DQN 算法：在计算目标 Q 值时引入合法动作掩码。
    """

    def __init__(self, model, optimizer, target_model, device=None, logger=None, monitor=None):
        """Initialize the algorithm.

        初始化算法。
        """
        self.model = model
        self.optimizer = optimizer
        self.target_model = target_model
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.gamma = Config.GAMMA
        self.tau = Config.TAU
        self.train_step = 0

    def learn(self, list_sample_data):
        """Perform one Double DQN gradient step.

        执行一步 Double DQN 梯度更新。
        """
        if not list_sample_data:
            return {}

        # 1) Prepare batch tensors
        # 准备批处理张量
        obs = torch.stack([torch.as_tensor(s.obs) for s in list_sample_data]).to(self.device)
        act = torch.stack([torch.as_tensor(s.act) for s in list_sample_data]).to(self.device).long().view(-1, 1)
        rew = torch.stack([torch.as_tensor(s.reward) for s in list_sample_data]).to(self.device).view(-1, 1)
        next_obs = torch.stack([torch.as_tensor(s.next_obs) for s in list_sample_data]).to(self.device)
        next_legal = torch.stack([torch.as_tensor(s.next_legal_action) for s in list_sample_data]).to(self.device)
        done = torch.stack([torch.as_tensor(s.done) for s in list_sample_data]).to(self.device).view(-1, 1)

        self.model.set_train_mode()
        
        # 2) Compute current Q values
        # 计算当前 Q 值
        current_q = self.model(obs).gather(1, act)

        # 3) Compute target Q values using Double DQN logic
        # 使用 Double DQN 逻辑计算目标 Q 值
        with torch.no_grad():
            # Selection: Use online model to select best next action among legal ones
            # 选择：使用在线模型从合法动作中选出最优下一步动作
            next_q_online = self.model(next_obs)
            next_q_online = next_q_online.masked_fill(next_legal < 0.5, -1e9)
            best_next_act = next_q_online.argmax(dim=1, keepdim=True)
            
            # Evaluation: Use target model to evaluate the selected action
            # 评估：使用目标模型评估选中的动作
            next_q_target_all = self.target_model(next_obs)
            next_q_target = next_q_target_all.gather(1, best_next_act)
            
            target_q = rew + self.gamma * next_q_target * (1.0 - done)

        # 4) Compute loss and update
        # 计算损失并更新
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE)
            
        self.optimizer.step()
        self.train_step += 1

        # 5) Soft update target network
        # 软更新目标网络
        self._soft_update_target()

        return {"loss": loss.item(), "q_mean": current_q.mean().item()}

    def _soft_update_target(self):
        """Soft update target network parameters: target = tau * online + (1-tau) * target.

        目标网络参数软更新。
        """
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
