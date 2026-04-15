#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

from collections import deque

import numpy as np

from agent_ppo.conf.conf import Config


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值线性归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """Feature preprocessor for Robot Vacuum.

    清扫大作战特征预处理器。
    """

    GRID_SIZE = 128
    VIEW_HALF = 10  # Full local view radius (21×21) / 完整局部视野半径
    LOCAL_HALF = 3  # Cropped view radius (7×7) / 裁剪后的视野半径
    MAX_BFS_DIST = 256
    HISTORY_LEN = 20

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state at episode start.

        对局开始时重置所有状态。
        """
        self.step_no = 0
        self.battery = 600
        self.battery_max = 600

        self.cur_pos = (0, 0)

        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        # Global passable map (0=obstacle, 1=passable), used for ray computation
        # 维护全局通行地图（0=障碍, 1=可通行），用于射线计算
        self.passable_map = np.ones((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.explored_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.dirty_memory_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        # Nearest dirt distance
        # 最近污渍距离
        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0

        self.last_nearest_npc_dist = 200.0
        self.nearest_npc_dist = 200.0

        self.charger_positions = []
        self.npc_positions = []
        self.last_npc_positions = []
        self.npc_velocities = []

        self._last_bfs_dist = float(self.MAX_BFS_DIST)
        self._last_bfs_targets_key = None
        self._last_bfs_step = -1

        self.position_history = deque(maxlen=self.HISTORY_LEN)

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

    def _extract_positions(self, data):
        """Extract (x, z) positions recursively from dict/list structures.

        递归提取 (x, z) 坐标。
        """
        positions = []
        if data is None:
            return positions

        if isinstance(data, dict):
            if "x" in data and ("z" in data or "y" in data):
                x = int(data.get("x", 0))
                z = int(data.get("z", data.get("y", 0)))
                return [(x, z)]

            pos = data.get("pos")
            if isinstance(pos, dict):
                positions.extend(self._extract_positions(pos))

            for v in data.values():
                if isinstance(v, (dict, list, tuple)):
                    positions.extend(self._extract_positions(v))

        elif isinstance(data, (list, tuple)):
            for item in data:
                positions.extend(self._extract_positions(item))

        return positions

    def _dedupe_positions(self, positions):
        uniq = []
        seen = set()
        for x, z in positions:
            if 0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE:
                key = (int(x), int(z))
                if key not in seen:
                    seen.add(key)
                    uniq.append(key)
        return uniq

    def _extract_entities_by_keys(self, data_dict, include_keywords):
        """Extract positions from keys containing any keyword.

        从包含关键字的字段提取坐标。
        """
        if not isinstance(data_dict, dict):
            return []
        positions = []
        for key, value in data_dict.items():
            key_l = str(key).lower()
            if any(k in key_l for k in include_keywords):
                positions.extend(self._extract_positions(value))
        return self._dedupe_positions(positions)

    def _extract_entities_by_exact_keys(self, data_dict, key_candidates):
        """Extract positions from exact key names (case-insensitive).

        从明确字段名提取坐标（大小写不敏感）。
        """
        if not isinstance(data_dict, dict):
            return []

        lower_map = {str(k).lower(): v for k, v in data_dict.items()}
        positions = []
        for key in key_candidates:
            value = lower_map.get(str(key).lower())
            if value is not None:
                positions.extend(self._extract_positions(value))

        return self._dedupe_positions(positions)

    def _parse_charger_positions(self, env_info):
        chargers = self._extract_entities_by_exact_keys(env_info, Config.CHARGER_PRIMARY_KEYS)
        if chargers:
            return chargers
        return self._extract_entities_by_keys(env_info, include_keywords=["charger", "charge"])

    def _parse_npc_positions(self, frame_state, env_info):
        # NPC positions are usually in env_info but some environments may also include robot lists in frame_state.
        # NPC 位置通常在 env_info，部分环境也可能在 frame_state 中提供机器人列表。
        from_env = self._extract_entities_by_exact_keys(env_info, Config.NPC_PRIMARY_KEYS)
        from_frame = self._extract_entities_by_exact_keys(frame_state, Config.NPC_PRIMARY_KEYS)

        if not from_env and not from_frame:
            from_env = self._extract_entities_by_keys(env_info, include_keywords=["robot", "npc", "enemy", "opponent"])
            from_frame = self._extract_entities_by_keys(
                frame_state,
                include_keywords=["robot", "npc", "enemy", "opponent"],
            )

        merged = self._dedupe_positions(from_env + from_frame)
        merged = [p for p in merged if p != self.cur_pos]
        return merged

    def _estimate_npc_velocities(self):
        """Estimate NPC velocities by nearest matching to previous frame.

        用与上一帧最近匹配的方法估计 NPC 速度。
        """
        velocities = []
        prev = list(self.last_npc_positions)
        used = set()
        for cx, cz in self.npc_positions:
            best_i = -1
            best_d = 1e18
            for i, (px, pz) in enumerate(prev):
                if i in used:
                    continue
                d = (cx - px) ** 2 + (cz - pz) ** 2
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0:
                used.add(best_i)
                px, pz = prev[best_i]
                velocities.append((cx - px, cz - pz))
            else:
                velocities.append((0.0, 0.0))
        return velocities

    def pb2struct(self, env_obs, last_action):
        """Parse and cache essential fields from observation dict.

        从 env_obs 字典中提取并缓存所有需要的状态量。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))
        self.position_history.append(self.cur_pos)

        self.last_npc_positions = list(self.npc_positions)
        self.charger_positions = self._parse_charger_positions(env_info)
        self.npc_positions = self._parse_npc_positions(frame_state, env_info)
        self.npc_velocities = self._estimate_npc_velocities()

        # Battery / 电量
        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        # Cleaning progress / 清扫进度
        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        # Legal actions / 合法动作
        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

    def _update_passable(self, hx, hz):
        """Write local view into global passable map.

        将局部视野写入全局通行地图。
        """
        view = self._view_map
        vsize = view.shape[0]
        half = vsize // 2

        for ri in range(vsize):
            for ci in range(vsize):
                gx = hx - half + ri
                gz = hz - half + ci
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    # 0 = obstacle, 1/2 = passable
                    # 0 = 障碍, 1/2 = 可通行
                    cell = int(view[ri, ci])
                    self.passable_map[gx, gz] = 1 if cell != 0 else 0
                    self.explored_map[gx, gz] = 1
                    self.dirty_memory_map[gx, gz] = 1 if cell == 2 else 0

    def _nearest_position(self, positions):
        if not positions:
            return None, 1e9
        hx, hz = self.cur_pos
        best_pos = None
        best_dist = 1e9
        for x, z in positions:
            d = float(np.sqrt((x - hx) ** 2 + (z - hz) ** 2))
            if d < best_dist:
                best_dist = d
                best_pos = (x, z)
        return best_pos, best_dist

    def _bfs_shortest_path(self, start, targets):
        """Compute shortest path distance on known passable map.

        在已知通行图上计算最短路径距离。
        """
        if not targets:
            return self.MAX_BFS_DIST
        sx, sz = start
        if (sx, sz) in targets:
            return 0

        q = deque([(sx, sz, 0)])
        visited = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        visited[sx, sz] = 1

        while q:
            x, z, dist = q.popleft()
            if dist >= self.MAX_BFS_DIST:
                continue

            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, nz = x + dx, z + dz
                if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                    continue
                if visited[nx, nz] == 1:
                    continue
                if self.passable_map[nx, nz] == 0:
                    continue

                nd = dist + 1
                if (nx, nz) in targets:
                    return nd

                visited[nx, nz] = 1
                q.append((nx, nz, nd))

        return self.MAX_BFS_DIST

    def _cached_bfs_shortest_path(self, start, targets):
        """Cache BFS result and update at a lower frequency to reduce overhead.

        缓存 BFS 结果并降低更新频率，减少计算开销。
        """
        if not targets:
            return self.MAX_BFS_DIST

        targets_key = tuple(sorted(targets))
        interval = max(int(Config.BFS_RECOMPUTE_INTERVAL), 1)

        if (
            self._last_bfs_targets_key == targets_key
            and self._last_bfs_step >= 0
            and (self.step_no - self._last_bfs_step) < interval
        ):
            return self._last_bfs_dist

        dist = self._bfs_shortest_path(start, targets)
        self._last_bfs_dist = float(dist)
        self._last_bfs_targets_key = targets_key
        self._last_bfs_step = int(self.step_no)
        return dist

    def _get_charger_feature(self):
        """Charger feature (4D): nearest charger xy + Euclidean/BFS distance.

        充电桩特征（4D）：最近充电桩坐标 + 欧氏距离/BFS距离。
        """
        nearest, euc_dist = self._nearest_position(self.charger_positions)
        if nearest is None:
            return np.zeros(Config.CHARGER_FEATURE_LEN, dtype=np.float32)

        cx, cz = nearest
        bfs_dist = self._cached_bfs_shortest_path(self.cur_pos, set(self.charger_positions))
        return np.array(
            [
                _norm(cx, self.GRID_SIZE - 1),
                _norm(cz, self.GRID_SIZE - 1),
                _norm(euc_dist, 180),
                _norm(bfs_dist, self.MAX_BFS_DIST),
            ],
            dtype=np.float32,
        )

    def _direction_one_hot(self, dx, dz, bins=8):
        one_hot = np.zeros(bins, dtype=np.float32)
        if dx == 0 and dz == 0:
            return one_hot
        angle = float(np.arctan2(dz, dx))
        idx = int(np.floor((angle + np.pi) / (2 * np.pi) * bins)) % bins
        one_hot[idx] = 1.0
        return one_hot

    def _get_npc_feature(self):
        """NPC feature: nearest NPC relative state + danger approach direction.

        NPC 特征：最近 NPC 相对状态 + 危险接近方向。
        """
        feat = np.zeros(Config.NPC_FEATURE_LEN, dtype=np.float32)
        if not self.npc_positions:
            return feat

        hx, hz = self.cur_pos
        best_idx = 0
        best_dist = 1e9
        for i, (nx, nz) in enumerate(self.npc_positions):
            d = float(np.sqrt((nx - hx) ** 2 + (nz - hz) ** 2))
            if d < best_dist:
                best_dist = d
                best_idx = i

        nx, nz = self.npc_positions[best_idx]
        vx, vz = self.npc_velocities[best_idx] if best_idx < len(self.npc_velocities) else (0.0, 0.0)

        dx = nx - hx
        dz = nz - hz

        self.last_nearest_npc_dist = self.nearest_npc_dist
        self.nearest_npc_dist = best_dist

        # Positive value indicates approaching toward agent.
        # 值越大表示 NPC 越朝向智能体接近。
        to_agent_x = hx - nx
        to_agent_z = hz - nz
        denom = float(np.sqrt(to_agent_x * to_agent_x + to_agent_z * to_agent_z) + 1e-6)
        approach_speed = (vx * to_agent_x + vz * to_agent_z) / denom

        danger_flag = 1.0 if (best_dist <= 8.0 and approach_speed > 0.0) else 0.0
        danger_dir = self._direction_one_hot(dx, dz, bins=Config.NPC_DIR_BINS) * danger_flag
        vel_dir = self._direction_one_hot(vx, vz, bins=Config.NPC_VEL_DIR_BINS)

        base = np.array(
            [
                float(np.clip((dx + self.GRID_SIZE) / (2 * self.GRID_SIZE), 0.0, 1.0)),
                float(np.clip((dz + self.GRID_SIZE) / (2 * self.GRID_SIZE), 0.0, 1.0)),
                _norm(best_dist, 180),
                float(np.clip((approach_speed + 2.0) / 4.0, 0.0, 1.0)),
                danger_flag,
            ],
            dtype=np.float32,
        )
        feat[:5] = base
        danger_end = 5 + Config.NPC_DIR_BINS
        feat[5:danger_end] = danger_dir
        feat[danger_end:] = vel_dir
        return feat

    def _get_trajectory_feature(self):
        """Trajectory feature (4D): revisit/backtrack/turn/progress metrics.

        轨迹特征（4D）：回访、折返、转向、推进效率。
        """
        hist = list(self.position_history)
        if len(hist) < 2:
            return np.zeros(Config.TRAJECTORY_FEATURE_LEN, dtype=np.float32)

        unique_ratio = len(set(hist)) / float(len(hist))
        revisit_ratio = 1.0 - unique_ratio

        backtrack = 1.0 if len(hist) >= 3 and hist[-1] == hist[-3] else 0.0

        moves = []
        for i in range(1, len(hist)):
            dx = hist[i][0] - hist[i - 1][0]
            dz = hist[i][1] - hist[i - 1][1]
            moves.append((dx, dz))

        turns = 0
        valid_turns = 0
        for i in range(1, len(moves)):
            if moves[i - 1] == (0, 0) or moves[i] == (0, 0):
                continue
            valid_turns += 1
            if moves[i - 1] != moves[i]:
                turns += 1
        turn_rate = turns / float(max(valid_turns, 1))

        disp = float(np.sqrt((hist[-1][0] - hist[0][0]) ** 2 + (hist[-1][1] - hist[0][1]) ** 2))
        max_disp = float((len(hist) - 1) * np.sqrt(2.0))
        progress_eff = float(np.clip(disp / max(max_disp, 1e-6), 0.0, 1.0))

        return np.array([revisit_ratio, backtrack, turn_rate, progress_eff], dtype=np.float32)

    def _get_global_dirt_map_feature(self):
        """Global dirt map feature: 16x16 coarse dirt ratio on explored cells.

        全局污渍图特征：16x16 粗粒度块内污渍比例（仅统计已探索格）。
        """
        size = Config.DIRT_MAP_SIZE
        block = self.GRID_SIZE // size

        explored = self.explored_map[: size * block, : size * block]
        dirty = self.dirty_memory_map[: size * block, : size * block]

        explored_cnt = explored.reshape(size, block, size, block).sum(axis=(1, 3)).astype(np.float32)
        dirty_cnt = dirty.reshape(size, block, size, block).sum(axis=(1, 3)).astype(np.float32)

        ratio = np.zeros((size, size), dtype=np.float32)
        valid = explored_cnt > 0
        ratio[valid] = dirty_cnt[valid] / explored_cnt[valid]
        return ratio.flatten().astype(np.float32)

    def _get_map_memory_feature(self):
        """Map memory feature: global explored/dirty bitmap in coarse grids.

        地图记忆特征：粗粒度网格的已探索/未清扫全局 bitmap。
        """
        size = Config.MAP_MEMORY_SIZE
        block = self.GRID_SIZE // size

        explored = self.explored_map[: size * block, : size * block]
        dirty = self.dirty_memory_map[: size * block, : size * block]

        explored_grid = explored.reshape(size, block, size, block).mean(axis=(1, 3)).astype(np.float32)
        dirty_grid = dirty.reshape(size, block, size, block).mean(axis=(1, 3)).astype(np.float32)

        return np.concatenate([explored_grid.flatten(), dirty_grid.flatten()], dtype=np.float32)

    def _get_local_view_feature(self):
        """Local view feature (49D): crop center 7×7 from 21×21.

        局部视野特征（49D）：从 21×21 视野中心裁剪 7×7。
        """
        center = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[center - h : center + h + 1, center - h : center + h + 1]
        return (crop / 2.0).flatten()

    def _get_global_state_feature(self):
        """Global state feature with extended engineering features.

        包含扩展特征工程的全局状态特征。

        Dimensions / 维度说明：
          [0]  step_norm         step progress / 步数归一化 [0,1]
          [1]  battery_ratio     battery level / 电量比 [0,1]
          [2]  cleaning_progress cleaned ratio / 已清扫比例 [0,1]
          [3]  remaining_dirt    remaining dirt ratio / 剩余污渍比例 [0,1]
          [4]  pos_x_norm        x position / x 坐标归一化 [0,1]
          [5]  pos_z_norm        z position / z 坐标归一化 [0,1]
          [6]  ray_N_dirt        north ray distance / 向上（z-）方向最近污渍距离
          [7]  ray_E_dirt        east ray distance / 向右（x+）方向
          [8]  ray_S_dirt        south ray distance / 向下（z+）方向
          [9]  ray_W_dirt        west ray distance / 向左（x-）方向
          [10] nearest_dirt_norm nearest dirt Euclidean distance / 最近污渍欧氏距离归一化
          [11] dirt_delta        approaching dirt indicator / 是否在接近污渍（1=是, 0=否）
        """
        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress

        hx, hz = self.cur_pos
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        # 4-directional ray to find nearest dirt
        # 四方向射线找最近污渍距离
        ray_dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # N E S W
        ray_dirt = []
        max_ray = 30
        for dx, dz in ray_dirs:
            x, z = hx, hz
            found = max_ray
            for step in range(1, max_ray + 1):
                x += dx
                z += dz
                if not (0 <= x < self.GRID_SIZE and 0 <= z < self.GRID_SIZE):
                    break
                if self._view_map is not None:
                    cell = (
                        int(
                            self._view_map[
                                np.clip(x - (hx - self.VIEW_HALF), 0, 20), np.clip(z - (hz - self.VIEW_HALF), 0, 20)
                            ]
                        )
                        if (0 <= x - hx + self.VIEW_HALF < 21 and 0 <= z - hz + self.VIEW_HALF < 21)
                        else 0
                    )
                    if cell == 2:
                        found = step
                        break
            ray_dirt.append(_norm(found, max_ray))

        # Nearest dirt Euclidean distance (estimated from 7×7 crop)
        # 最近污渍欧氏距离（视野内 7×7 粗估）
        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self.nearest_dirt_dist, 180)

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0

        base_global = np.array(
            [
                step_norm,
                battery_ratio,
                cleaning_progress,
                remaining_dirt,
                pos_x_norm,
                pos_z_norm,
                ray_dirt[0],
                ray_dirt[1],
                ray_dirt[2],
                ray_dirt[3],
                nearest_dirt_norm,
                dirt_delta,
            ],
            dtype=np.float32,
        )

        charger_feat = self._get_charger_feature()
        npc_feat = self._get_npc_feature()
        traj_feat = self._get_trajectory_feature()
        dirt_map_feat = self._get_global_dirt_map_feature()
        map_memory_feat = self._get_map_memory_feature()

        return np.concatenate(
            [base_global, charger_feat, npc_feat, traj_feat, dirt_map_feat, map_memory_feat],
            dtype=np.float32,
        )

    def _calc_nearest_dirt_dist(self):
        """Find nearest dirt Euclidean distance from local view.

        从局部视野中找最近污渍的欧氏距离。
        """
        view = self._view_map
        if view is None:
            return 200.0
        dirt_coords = np.argwhere(view == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((dirt_coords[:, 0] - center) ** 2 + (dirt_coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def get_legal_action(self):
        """Return legal action mask (8D list).

        返回合法动作掩码（8D list）。
        """
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        """Generate feature vector, legal action mask, and scalar reward.

        生成特征向量、合法动作掩码和标量奖励。
        """
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()  # 49D
        global_state = self._get_global_state_feature()
        legal_action = self.get_legal_action()  # 8D
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr]).astype(np.float32)

        reward = self.reward_process()

        return feature, legal_action, reward

    def reward_process(self):
        # Cleaning reward / 清扫奖励
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_reward = 0.1 * cleaned_this_step

        # Step penalty / 时间惩罚
        step_penalty = -0.001

        return cleaning_reward + step_penalty
