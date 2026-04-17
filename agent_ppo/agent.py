#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
清扫大作战 Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（特征向量 + 合法动作掩码）。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 中取出动作整数并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时推理（随机采样动作）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)

        legal_arr = self._sanitize_legal_action(legal_action)
        prob = self._legal_soft_max(logits, legal_arr)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def _sanitize_legal_action(self, legal_action):
        """Convert legal action mask to 8D float array and ensure at least one legal action.

        将合法动作掩码转为 8 维 float 数组，并保证至少有一个合法动作。
        """
        try:
            arr = np.asarray(legal_action, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            arr = np.ones(Config.ACTION_NUM, dtype=np.float32)

        if arr.size < Config.ACTION_NUM:
            pad = np.ones(Config.ACTION_NUM - arr.size, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.size > Config.ACTION_NUM:
            arr = arr[: Config.ACTION_NUM]

        mask = (arr > 0.5).astype(np.float32)
        if float(mask.sum()) <= 0.0:
            mask[:] = 1.0
        return mask

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时推理（贪心）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行训练。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature):
        """Gradient-free forward pass, returns (logits_np, value_np).

        无梯度推理，返回 (logits_np, value_np)。
        """
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        legal = self._sanitize_legal_action(legal_action)
        masked_logits = np.where(legal > 0.5, logits, -1e9)
        max_logit = float(np.max(masked_logits))
        exp_logits = np.exp(np.clip(masked_logits - max_logit, -50.0, 50.0)) * legal

        denom = float(np.sum(exp_logits))
        if (not np.isfinite(denom)) or denom <= 0.0:
            exp_logits = legal.astype(np.float64)
            denom = float(np.sum(exp_logits))

        return (exp_logits / max(denom, 1e-9)).astype(np.float32)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        if probs.size < Config.ACTION_NUM:
            probs = np.pad(probs, (0, Config.ACTION_NUM - probs.size), mode="constant", constant_values=0.0)
        elif probs.size > Config.ACTION_NUM:
            probs = probs[: Config.ACTION_NUM]

        probs = np.clip(probs, 0.0, None)
        p_sum = float(np.sum(probs))
        if (not np.isfinite(p_sum)) or p_sum <= 0.0:
            probs = np.ones(Config.ACTION_NUM, dtype=np.float64) / float(Config.ACTION_NUM)
        else:
            probs = probs / p_sum

        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(Config.ACTION_NUM, p=probs))
