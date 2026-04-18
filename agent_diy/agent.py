#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY DQN Agent.
清扫大作战 DIY DQN Agent 实现。
"""

import random
import torch
import numpy as np

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model
from kaiwudrl.interface.agent import BaseAgent

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    """DIY DQN Agent with legal action masking.

    DQN 智能体：支持 epsilon-greedy 采样和合法动作掩码。
    """

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device
        self.model = Model(device).to(self.device)
        self.target_model = Model(device).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=Config.LEARNING_RATE)
        self.algorithm = Algorithm(
            self.model, self.optimizer, self.target_model, self.device, logger, monitor
        )
        self.preprocessor = Preprocessor()

        self.last_action = -1
        self.steps_done = 0
        self.epsilon = Config.EPSILON_START

        # Hard unstuck state
        # 硬脱困状态维护
        self._same_pos_steps = 0
        self._last_pos = None
        self._hard_ban_action = -1
        self._hard_ban_left = 0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset internal state at episode start.

        对局开始重置。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self._same_pos_steps = 0
        self._last_pos = None
        self._hard_ban_action = -1
        self._hard_ban_left = 0

    def observation_process(self, env_obs):
        """Convert environment observation to model feature.

        特征处理：原始观测 -> 特征向量。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(
            env_obs, self.last_action
        )
        self._update_hard_unstuck_state()

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Extract action from ActData.

        动作处理：ActData -> 环境动作。
        """
        action = act_data.action
        self.last_action = int(action[0])

        if self._hard_ban_left > 0:
            self._hard_ban_left -= 1

        return self.last_action

    def predict(self, list_obs_data):
        """Epsilon-greedy inference for training.

        训练推理（Epsilon-greedy 采样）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        # Update epsilon
        # 更新探索率
        self.steps_done += 1
        self.epsilon = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * np.exp(
            -1.0 * self.steps_done / Config.EPSILON_DECAY
        )

        q_values = self._run_model(feature)

        # Apply legal mask to prevent getting stuck
        # 应用合法动作掩码，防止在角落原地打转
        legal_arr = self._sanitize_legal_action(legal_action)
        legal_arr = self._apply_hard_unstuck_ban(legal_arr)

        if float(np.sum(legal_arr)) <= 0.0:
            # Fallback if no legal actions
            legal_arr = np.ones(Config.ACTION_NUM, dtype=np.float32)

        if random.random() < self.epsilon:
            # Exploration: choose random action from legal set
            legal_indices = np.where(legal_arr > 0.5)[0]
            action = int(random.choice(legal_indices))
        else:
            # Exploitation: choose max Q action among legal set
            masked_q = np.where(legal_arr > 0.5, q_values, -1e9)
            action = int(np.argmax(masked_q))

        return [ActData(action=[action], prob=list(q_values))]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估推理（纯贪心）。
        """
        obs_data, _ = self.observation_process(env_obs)
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        q_values = self._run_model(feature)

        legal_arr = self._sanitize_legal_action(legal_action)
        legal_arr = self._apply_hard_unstuck_ban(legal_arr)
        if float(np.sum(legal_arr)) <= 0.0:
            legal_arr = np.ones(Config.ACTION_NUM, dtype=np.float32)

        masked_q = np.where(legal_arr > 0.5, q_values, -1e9)
        action = int(np.argmax(masked_q))

        return self.action_process(ActData(action=[action]))

    def learn(self, list_sample_data):
        """Train the model.

        训练入口。
        """
        return self.algorithm.learn(list_sample_data)

    def _run_model(self, feature):
        self.model.set_eval_mode()
        obs_tensor = torch.as_tensor(np.array([feature], dtype=np.float32)).to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
            
        with torch.no_grad():
            q_values = self.model(obs_tensor)
        return q_values.cpu().numpy()[0]

    def _sanitize_legal_action(self, legal_action):
        try:
            arr = np.asarray(legal_action, dtype=np.float32).reshape(-1)
        except:
            arr = np.ones(Config.ACTION_NUM, dtype=np.float32)

        if arr.size < Config.ACTION_NUM:
            arr = np.pad(arr, (0, Config.ACTION_NUM - arr.size), constant_values=1.0)
        return (arr > 0.5).astype(np.float32)

    def _update_hard_unstuck_state(self):
        if not Config.ENABLE_HARD_UNSTUCK:
            return
        cur_pos = self.preprocessor.cur_pos
        if self._last_pos == cur_pos:
            self._same_pos_steps += 1
        else:
            self._same_pos_steps = 0
        self._last_pos = cur_pos

        if self._same_pos_steps >= Config.HARD_UNSTUCK_TRIGGER_STEPS:
            if self._hard_ban_left <= 0:
                self._hard_ban_action = self.last_action
                self._hard_ban_left = Config.HARD_UNSTUCK_BAN_STEPS

    def _apply_hard_unstuck_ban(self, legal_arr):
        if self._hard_ban_left <= 0 or self._hard_ban_action < 0:
            return legal_arr
        out = legal_arr.copy()
        out[int(self._hard_ban_action)] = 0.0
        return out

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())
        self.logger.info(f"load model {model_file_path} successfully")
