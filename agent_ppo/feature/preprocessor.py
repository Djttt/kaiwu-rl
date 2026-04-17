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
    ACTION_DIRS = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

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

    def _sanitize_legal_action(self, legal_action):
        """Normalize legal action mask to 8D 0/1 list.

        将合法动作掩码标准化为 8 维 0/1 列表。
        """
        if legal_action is None:
            return [1] * Config.ACTION_NUM

        try:
            arr = list(legal_action)
        except TypeError:
            return [1] * Config.ACTION_NUM

        out = [1] * Config.ACTION_NUM
        for i in range(Config.ACTION_NUM):
            if i < len(arr):
                try:
                    out[i] = 1 if float(arr[i]) > 0.5 else 0
                except (TypeError, ValueError):
                    out[i] = 0
        return out

    def _is_passable_in_view(self, ix, iz):
        if self._view_map is None:
            return True
        h, w = self._view_map.shape
        if ix < 0 or ix >= h or iz < 0 or iz >= w:
            return False
        return int(self._view_map[ix, iz]) != 0

    def _compute_local_legal_action(self):
        """Build legal action mask from local map with anti-corner rule.

        基于局部地图和防穿角规则构造合法动作掩码。
        """
        if self._view_map is None:
            return [1] * Config.ACTION_NUM

        center = self._view_map.shape[0] // 2
        legal = [0] * Config.ACTION_NUM

        for i, (dx, dz) in enumerate(self.ACTION_DIRS):
            tx, tz = center + dx, center + dz
            if not self._is_passable_in_view(tx, tz):
                continue

            if dx != 0 and dz != 0:
                # Diagonal anti-corner: at least one side-adjacent cell must be passable.
                # 斜向防穿角：至少有一条边邻格可通行。
                side_x = self._is_passable_in_view(center + dx, center)
                side_z = self._is_passable_in_view(center, center + dz)
                if not (side_x or side_z):
                    continue

            legal[i] = 1

        return legal

    def _merge_legal_action(self, env_legal, local_legal):
        """Merge env legal action and local-map legal action robustly.

        稳健融合环境合法动作与局部地图合法动作。
        """
        merged = [int(e and l) for e, l in zip(env_legal, local_legal)]
        if sum(merged) > 0:
            return merged
        if sum(local_legal) > 0:
            return local_legal
        if sum(env_legal) > 0:
            return env_legal
        # Keep all-zero mask to trigger upper-level safe fallback instead of pretending all moves are legal.
        # 保持全0掩码，让上层走安全回退，而不是伪造“全合法”。
        return [0] * Config.ACTION_NUM

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
        # Compatible with multiple field names from different environment versions.
        # 兼容不同环境版本中的合法动作字段命名。
        env_legal = self._sanitize_legal_action(
            observation.get("legal_action", observation.get("legal_act", observation.get("legal_actions")))
        )

        # Local view map (21×21) / 局部视野地图
        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.float32)
            hx, hz = self.cur_pos
            self._update_passable(hx, hz)

        local_legal = self._compute_local_legal_action()
        self._legal_act = self._merge_legal_action(env_legal, local_legal)

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

    def _get_zone_target_center(self):
        """Get the best coarse cleaning zone center from dirty/explored memory.

        从污渍/探索记忆中选取最佳粗粒度清扫分区中心。
        """
        mid = self.GRID_SIZE // 2
        zones = [
            (0, mid, 0, mid),
            (0, mid, mid, self.GRID_SIZE),
            (mid, self.GRID_SIZE, 0, mid),
            (mid, self.GRID_SIZE, mid, self.GRID_SIZE),
        ]
        centers = [
            (mid // 2, mid // 2),
            (mid // 2, (mid + self.GRID_SIZE) // 2),
            ((mid + self.GRID_SIZE) // 2, mid // 2),
            ((mid + self.GRID_SIZE) // 2, (mid + self.GRID_SIZE) // 2),
        ]

        best_idx = 0
        best_score = -1e9
        for i, (x0, x1, z0, z1) in enumerate(zones):
            dirty_density = float(np.mean(self.dirty_memory_map[x0:x1, z0:z1]))
            explored_density = float(np.mean(self.explored_map[x0:x1, z0:z1]))
            unexplored_density = 1.0 - explored_density
            score = dirty_density + 0.35 * unexplored_density
            if score > best_score:
                best_score = score
                best_idx = i

        return centers[best_idx], float(best_score)

    def _window_memory_stats(self, x, z, radius=2):
        x0 = max(0, x - radius)
        x1 = min(self.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(self.GRID_SIZE, z + radius + 1)

        if x1 <= x0 or z1 <= z0:
            return 0.0, 0.0

        explored_density = float(np.mean(self.explored_map[x0:x1, z0:z1]))
        dirty_density = float(np.mean(self.dirty_memory_map[x0:x1, z0:z1]))
        unexplored_density = 1.0 - explored_density
        return dirty_density, unexplored_density

    def get_behavior_blend_alpha(self):
        """Return dynamic blend ratio between model policy and behavior strategy prior.

        返回模型策略与行为先验融合比例。
        """
        battery_ratio = float(self.battery) / float(max(self.battery_max, 1))
        low_battery = float(np.clip((0.45 - battery_ratio) / 0.45, 0.0, 1.0))

        npc_urgency = float(np.clip((8.0 - self.nearest_npc_dist) / 8.0, 0.0, 1.0))

        hist = list(self.position_history)
        revisit_risk = 0.0
        if len(hist) >= 8:
            revisit_risk = 1.0 - len(set(hist[-8:])) / float(len(hist[-8:]))

        alpha = 0.20 + 0.45 * max(low_battery, npc_urgency) + 0.20 * revisit_risk

        # If agent is continuously not moving, rely more on strategy prior to escape corners.
        # 若智能体连续原地不动，提升先验权重以脱困。
        if len(hist) >= 4 and len(set(hist[-4:])) == 1:
            alpha = max(alpha, 0.85)

        return float(np.clip(alpha, 0.20, 0.90))

    def get_action_strategy_prior(self, last_action=-1):
        """Build 8D action prior for charging, avoidance, and cleaning planning.

        构建 8 维动作先验：充电策略 + 躲避策略 + 清扫规划。
        """
        legal = np.asarray(self.get_legal_action(), dtype=np.float32).reshape(-1)
        if legal.size < Config.ACTION_NUM:
            legal = np.pad(legal, (0, Config.ACTION_NUM - legal.size), mode="constant", constant_values=1.0)
        elif legal.size > Config.ACTION_NUM:
            legal = legal[: Config.ACTION_NUM]

        # Prefer locally reconstructed legal actions to reduce wall-hitting when env mask is noisy.
        # 优先使用本地重建合法动作，降低环境掩码噪声导致的撞墙。
        local_legal = np.asarray(self._compute_local_legal_action(), dtype=np.float32).reshape(-1)
        if local_legal.size < Config.ACTION_NUM:
            local_legal = np.pad(local_legal, (0, Config.ACTION_NUM - local_legal.size), mode="constant")
        elif local_legal.size > Config.ACTION_NUM:
            local_legal = local_legal[: Config.ACTION_NUM]

        if float(np.sum(local_legal)) > 0.0:
            legal = legal * local_legal
            if float(np.sum(legal)) <= 0.0:
                legal = local_legal

        if float(np.sum(legal)) <= 0.0:
            legal = np.ones(Config.ACTION_NUM, dtype=np.float32)

        prior = np.ones(Config.ACTION_NUM, dtype=np.float32)
        hx, hz = self.cur_pos
        hist = list(self.position_history)
        recent_hist = set(hist[-8:]) if hist else set()

        battery_ratio = float(self.battery) / float(max(self.battery_max, 1))

        cur_charger_dist = 1e9
        if self.charger_positions:
            _, cur_charger_dist = self._nearest_position(self.charger_positions)

        target_center, _ = self._get_zone_target_center()

        for i, (dx, dz) in enumerate(self.ACTION_DIRS):
            if legal[i] < 0.5:
                prior[i] = 0.0
                continue

            nx, nz = hx + dx, hz + dz
            if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                prior[i] = 0.0
                continue

            # Cleaning planning: immediate local tile + memory frontier.
            # 清扫规划：邻格清扫价值 + 记忆前沿探索价值。
            lx = self.VIEW_HALF + dx
            lz = self.VIEW_HALF + dz
            if not self._is_passable_in_view(lx, lz):
                prior[i] = 0.0
                continue

            local_cell = int(self._view_map[lx, lz])
            if local_cell == 2:
                prior[i] *= 1.9
            elif local_cell == 1:
                prior[i] *= 0.95
            else:
                prior[i] *= 0.6

            dirty_density, unexplored_density = self._window_memory_stats(nx, nz, radius=2)
            prior[i] *= 1.0 + 0.65 * dirty_density + 0.35 * unexplored_density

            if (nx, nz) in recent_hist:
                prior[i] *= 0.55

            # Zone coverage planning when nearby dirt is sparse.
            # 附近污渍稀少时，按分区覆盖规划推进。
            if self.nearest_dirt_dist > 2.5:
                tx, tz = target_center
                vtx = float(tx - hx)
                vtz = float(tz - hz)
                tv_norm = float(np.sqrt(vtx * vtx + vtz * vtz) + 1e-6)
                align = float((dx * vtx + dz * vtz) / (np.sqrt(dx * dx + dz * dz) * tv_norm + 1e-6))
                prior[i] *= 1.0 + 0.25 * float(np.clip(align, -0.8, 1.0))

            # Charging strategy: reserve return energy and move toward charger when needed.
            # 充电策略：预留返程电量，低电时优先向充电桩推进。
            if self.charger_positions:
                next_charger_dist = min(float(np.hypot(nx - cx, nz - cz)) for cx, cz in self.charger_positions)
                improve = cur_charger_dist - next_charger_dist

                reserve_margin = 14.0 + 0.35 * cur_charger_dist
                battery_steps = float(self.battery)
                charge_urgency = float(np.clip((reserve_margin - battery_steps) / max(reserve_margin, 1.0), 0.0, 1.0))
                charge_urgency = max(charge_urgency, float(np.clip((0.42 - battery_ratio) / 0.42, 0.0, 1.0)))

                if charge_urgency > 0.0:
                    prior[i] *= 1.0 + (0.9 + 1.4 * charge_urgency) * float(np.clip(improve, -1.0, 1.5))

            # NPC avoidance: penalize directions entering predicted danger area.
            # 躲避策略：惩罚进入 NPC 预测危险区的方向。
            if self.npc_positions:
                risk = 0.0
                for j, (px, pz) in enumerate(self.npc_positions):
                    vx, vz = self.npc_velocities[j] if j < len(self.npc_velocities) else (0.0, 0.0)
                    pred_x = float(px + vx)
                    pred_z = float(pz + vz)
                    d = float(np.hypot(nx - pred_x, nz - pred_z))
                    risk += float(np.exp(-d / 3.0))
                    if d < 1.5:
                        risk += 2.0

                danger_scale = 1.6 if self.nearest_npc_dist < 8.0 else 1.0
                prior[i] *= float(np.exp(-0.45 * danger_scale * risk))

            # Spiral-like smooth movement: prefer straight/slight-turn, avoid hard backtrack.
            # 螺旋式平滑推进：偏好直行/小转向，抑制大折返。
            if last_action is not None and int(last_action) >= 0:
                diff = (i - int(last_action)) % Config.ACTION_NUM
                circular = min(diff, Config.ACTION_NUM - diff)
                if circular == 0:
                    prior[i] *= 1.08
                elif circular == 1:
                    prior[i] *= 1.12
                elif circular >= 3:
                    prior[i] *= 0.88
                if circular == 4:
                    prior[i] *= 0.65

        prior = np.clip(prior, 1e-5, 50.0) * legal
        if float(np.sum(prior)) <= 0.0:
            prior = legal.copy()
        if float(np.sum(prior)) <= 0.0:
            prior = np.ones(Config.ACTION_NUM, dtype=np.float32)
        return prior.astype(np.float32)

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
