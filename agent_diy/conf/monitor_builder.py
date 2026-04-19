#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Robot Vacuum.
清扫大作战监控面板配置构建器。
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    This function is used to create monitoring panel configurations for custom indicators.
    该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("扫地机器人-DQN")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="avg(reward{})",
        )
        .end_panel()
        .add_panel(
            name="DQN 训练损失",
            name_en="loss",
            type="line",
        )
        .add_metric(
            metrics_name="loss",
            expr="avg(loss{})",
        )
        .end_panel()
        .add_panel(
            name="平均 Q 值",
            name_en="q_mean",
            type="line",
        )
        .add_metric(
            metrics_name="q_mean",
            expr="avg(q_mean{})",
        )
        .end_panel()
        .add_panel(
            name="探索率 (Epsilon)",
            name_en="epsilon",
            type="line",
        )
        .add_metric(
            metrics_name="epsilon",
            expr="avg(epsilon{})",
        )
        .end_panel()
        .add_panel(
            name="训练局数",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="max(episode_cnt{})",
        )
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
