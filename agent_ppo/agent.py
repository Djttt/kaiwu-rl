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
        self._same_pos_steps = 0
        self._last_pos = None
        self._hard_ban_action = -1
        self._hard_ban_left = 0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self._same_pos_steps = 0
        self._last_pos = None
        self._hard_ban_action = -1
        self._hard_ban_left = 0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（特征向量 + 合法动作掩码）。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward
        self._update_hard_unstuck_state()

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

        if self._hard_ban_left > 0:
            self._hard_ban_left -= 1
            if self._hard_ban_left <= 0:
                self._hard_ban_left = 0
                self._hard_ban_action = -1

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
        legal_arr = self._apply_hard_unstuck_ban(legal_arr)
        if float(np.sum(legal_arr)) <= 0.0:
            # Fallback to strategy-derived valid set when external legal mask is empty.
            # 当外部合法动作掩码异常全0时，回退到策略先验推导的可行动作集合。
            fallback_prior = self.preprocessor.get_action_strategy_prior(last_action=self.last_action)
            fallback_prior = np.asarray(fallback_prior, dtype=np.float32).reshape(-1)
            if fallback_prior.size < Config.ACTION_NUM:
                fallback_prior = np.pad(
                    fallback_prior,
                    (0, Config.ACTION_NUM - fallback_prior.size),
                    mode="constant",
                    constant_values=0.0,
                )
            elif fallback_prior.size > Config.ACTION_NUM:
                fallback_prior = fallback_prior[: Config.ACTION_NUM]

            legal_arr = (fallback_prior > 1e-6).astype(np.float32)
            legal_arr = self._apply_hard_unstuck_ban(legal_arr)
            if float(np.sum(legal_arr)) <= 0.0:
                legal_arr = np.ones(Config.ACTION_NUM, dtype=np.float32)
                legal_arr = self._apply_hard_unstuck_ban(legal_arr)
                if float(np.sum(legal_arr)) <= 0.0:
                    legal_arr = np.ones(Config.ACTION_NUM, dtype=np.float32)

        model_prob = self._legal_soft_max(logits, legal_arr)
        prob = self._blend_with_action_strategy(model_prob, legal_arr)
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

    def _blend_with_action_strategy(self, model_prob, legal_arr):
        """Blend learned policy with behavior strategy prior.

        融合模型策略与行为先验。
        """
        strategy_prior = self.preprocessor.get_action_strategy_prior(last_action=self.last_action)
        strategy_prior = np.asarray(strategy_prior, dtype=np.float32).reshape(-1)
        if strategy_prior.size < Config.ACTION_NUM:
            strategy_prior = np.pad(
                strategy_prior,
                (0, Config.ACTION_NUM - strategy_prior.size),
                mode="constant",
                constant_values=1.0,
            )
        elif strategy_prior.size > Config.ACTION_NUM:
            strategy_prior = strategy_prior[: Config.ACTION_NUM]

        strategy_prior = np.clip(strategy_prior, 0.0, None) * legal_arr
        prior_sum = float(np.sum(strategy_prior))
        if (not np.isfinite(prior_sum)) or prior_sum <= 0.0:
            return model_prob

        strategy_prob = strategy_prior / prior_sum

        alpha = float(self.preprocessor.get_behavior_blend_alpha())
        alpha = float(np.clip(alpha, 0.0, 0.9))

        mixed_prob = (1.0 - alpha) * model_prob + alpha * strategy_prob
        mixed_prob = mixed_prob * legal_arr

        mixed_sum = float(np.sum(mixed_prob))
        if (not np.isfinite(mixed_sum)) or mixed_sum <= 0.0:
            return model_prob

        return (mixed_prob / mixed_sum).astype(np.float32)

    def _update_hard_unstuck_state(self):
        """Update stuck counters and trigger hard action ban when needed.

        更新卡住计数，并在需要时触发硬动作封禁。
        """
        if not bool(getattr(Config, "ENABLE_HARD_UNSTUCK", False)):
            return

        cur_pos = tuple(self.preprocessor.cur_pos) if hasattr(self.preprocessor, "cur_pos") else None
        if cur_pos is None:
            return

        if self._last_pos is None:
            self._same_pos_steps = 0
        elif cur_pos == self._last_pos:
            self._same_pos_steps += 1
        else:
            self._same_pos_steps = 0
        self._last_pos = cur_pos

        trigger_steps = max(int(getattr(Config, "HARD_UNSTUCK_TRIGGER_STEPS", 4)), 1)
        if self._same_pos_steps >= trigger_steps and self.last_action is not None and int(self.last_action) >= 0:
            if self._hard_ban_left <= 0:
                self._hard_ban_action = int(self.last_action)
                self._hard_ban_left = max(int(getattr(Config, "HARD_UNSTUCK_BAN_STEPS", 2)), 1)

    def _apply_hard_unstuck_ban(self, legal_arr):
        """Mask out banned action for a few steps in hard unstuck mode.

        在硬脱困模式下短时屏蔽被封禁动作。
        """
        legal = np.asarray(legal_arr, dtype=np.float32).reshape(-1)
        if legal.size < Config.ACTION_NUM:
            legal = np.pad(legal, (0, Config.ACTION_NUM - legal.size), mode="constant", constant_values=0.0)
        elif legal.size > Config.ACTION_NUM:
            legal = legal[: Config.ACTION_NUM]

        if not bool(getattr(Config, "ENABLE_HARD_UNSTUCK", False)):
            return legal
        if self._hard_ban_left <= 0:
            return legal

        ban_idx = int(self._hard_ban_action)
        if ban_idx < 0 or ban_idx >= Config.ACTION_NUM:
            return legal

        out = legal.copy()
        out[ban_idx] = 0.0

        # If masking leads to all-zero, leave fallback handling to caller.
        # 若屏蔽后全0，则交由上层回退逻辑处理。
        return out

    def _sanitize_legal_action(self, legal_action):
        """Convert legal action mask to 8D float array.

        将合法动作掩码转为 8 维 float 数组。
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

        return (arr > 0.5).astype(np.float32)

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
