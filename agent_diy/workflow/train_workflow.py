#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum DIY DQN Training Workflow.
清扫大作战 DIY DQN 训练工作流实现。
"""

import os
import time
import collections
import random
import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, reward_shaping
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


class ReplayBuffer:
    """Simple experience replay buffer.

    简易经验回放池。
    """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_env_score_total(env_reward):
    if env_reward is None:
        return 0.0
    if isinstance(env_reward, dict):
        for key in ("reward", "score", "clean_score", "total_score"):
            if key in env_reward:
                return _to_float(env_reward.get(key), 0.0)
    return 0.0


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """DIY Workflow for DQN training.

    DQN 训练工作流：包含经验收集和向训练节点发送样本。
    """
    env, agent = envs[0], agents[0]

    # Read and validate configuration
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    replay_buffer = ReplayBuffer(Config.REPLAY_BUFFER_CAPACITY)
    last_save_model_time = time.time()
    last_metrics_time = time.time()
    episode_cnt = 0

    logger.info("Starting DQN Training Workflow...")

    while True:
        # 1) Reset environment
        env_obs = env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, logger):
            continue

        # Load latest model from learner at episode start
        agent.load_model(id="latest")
        agent.reset(env_obs)
        obs_data, _ = agent.observation_process(env_obs)

        done = False
        step = 0
        total_reward = 0.0
        last_env_score_total = 0.0
        episode_cnt += 1

        # 2) Episode loop
        while not done:
            # Inference
            act_data_list = agent.predict([obs_data])
            act_data = act_data_list[0]
            act = agent.action_process(act_data)

            # Step
            env_reward, env_obs = env.step(act)
            if handle_disaster_recovery(env_obs, logger):
                break

            next_obs_data, _ = agent.observation_process(env_obs)
            
            terminated = env_obs["terminated"]
            truncated = env_obs["truncated"]
            done = terminated or truncated
            step += 1

            # Reward Calculation
            env_score_total = _extract_env_score_total(env_reward)
            score_delta = max(0.0, env_score_total - last_env_score_total)
            last_env_score_total = env_score_total

            reward_scalar = reward_shaping(
                obs=obs_data.feature,
                _obs=next_obs_data.feature,
                state=env_reward,
                _state=env_obs.get("observation", {}).get("env_info", {}),
                score_delta=score_delta,
                done=done
            )
            total_reward += reward_scalar

            # 3) Store transition in Replay Buffer
            transition = SampleData(
                obs=np.array(obs_data.feature, dtype=np.float32),
                legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                act=np.array([act]),
                reward=np.array([reward_scalar], dtype=np.float32),
                next_obs=np.array(next_obs_data.feature, dtype=np.float32),
                next_legal_action=np.array(next_obs_data.legal_action, dtype=np.float32),
                done=np.array([float(done)], dtype=np.float32)
            )
            replay_buffer.push(transition)

            # 4) Send sampled batch to Learner
            # Modified for distributed mode: use send_sample_data instead of learn()
            if len(replay_buffer) >= Config.BATCH_SIZE:
                batch = replay_buffer.sample(Config.BATCH_SIZE)
                agent.send_sample_data(batch)

            obs_data = next_obs_data

        logger.info(f"Episode {episode_cnt} finished. Steps: {step}, Reward: {total_reward:.2f}")

        # 5) Periodic Tasks: Metrics, Monitoring, Saving
        now = time.time()
        if now - last_metrics_time >= 60:
            training_metrics = get_training_metrics()
            if training_metrics:
                logger.info(f"training_metrics: {training_metrics}")
            if monitor:
                monitor.put_data(
                    {
                        os.getpid(): {
                            "reward": total_reward,
                            "episode_cnt": episode_cnt,
                            "epsilon": getattr(agent, "epsilon", 1.0),
                        }
                    }
                )
            last_metrics_time = now

        # In distributed mode, learner usually handles saving, but aisrv can trigger it.
        if now - last_save_model_time >= 1800:
            agent.save_model()
            last_save_model_time = now
