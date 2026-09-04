"""
EV Charging Control environment (Gymnasium API) for DSCD 614 Option TD3-1.

Action:
  Normalized continuous charging fraction in [0.0, 1.0].
  Actual physical power = action * vehicle power limit.

Reward:
  - Dense reward for useful SOC progress toward target SOC.
  - Linear penalty for electricity cost drawn.
  - Quadratic penalty on action intensity scaled by grid demand (incentivizes smoothing).
  - Terminal bonus for meeting target SOC (>= target - 0.5%) or penalty for shortfall.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class EVChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_path,
        step_minutes=15,
        seed=None,
        w_progress=20.0,
        w_cost=2.0,
        w_peak=2.0,
        w_terminal_bonus=10.0,
        w_shortfall=20.0,
        split="all",
        test_frac=0.2,
    ):
        super().__init__()

        df = pd.read_csv(data_path)
        df["ts"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["ts"].dt.hour

        # Empirical background grid demand curve by hour of day
        self.hourly_demand = df.groupby("hour")["charging_demand"].mean().to_dict()

        self.step_minutes = step_minutes
        self.dt_hours = step_minutes / 60.0

        # Discretize session duration into 15-min budget
        df["n_steps"] = (
            np.ceil(df["charging_duration"] / step_minutes).clip(lower=1).astype(int)
        )

        df = df.sort_values("ts").reset_index(drop=True)
        df_full = df
        split_idx = int(len(df) * (1 - test_frac))
        if split == "train":
            df = df.iloc[:split_idx]
        elif split == "test":
            df = df.iloc[split_idx:]
        elif split != "all":
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")

        self.split = split
        self.sessions = df[
            [
                "initial_soc",
                "final_soc",
                "battery_capacity_kWh",
                "charging_power_kW",
                "electricity_price",
                "hour",
                "n_steps",
            ]
        ].reset_index(drop=True)

        self.max_power_global = float(df_full["charging_power_kW"].max())

        # Observation: [soc, target_soc, steps_remaining_frac, price_norm, demand_norm, power_limit_norm]
        self.observation_space = spaces.Box(
            low=0.0, high=1.5, shape=(6,), dtype=np.float32
        )

        # Normalized action: fraction of current EV's maximum power [0.0, 1.0]
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self._init_seed = seed
        self._seeded_once = False

        self.w_progress = w_progress
        self.w_cost = w_cost
        self.w_peak = w_peak
        self.w_terminal_bonus = w_terminal_bonus
        self.w_shortfall = w_shortfall

        self._state = None
        self._initial_hour = 0
        self._current_hour = 0
        self._n_steps = 0
        self._steps_remaining = 0
        self._step_count = 0

    def _get_obs(self):
        s = self._state
        demand = self.hourly_demand[self._current_hour % 24]
        return np.array(
            [
                s["soc"] / 100.0,
                s["target_soc"] / 100.0,
                self._steps_remaining / max(self._n_steps, 1),
                s["price"] / 15.0,
                demand / 100.0,
                s["power_limit"] / self.max_power_global,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        if seed is None and not self._seeded_once:
            seed = self._init_seed
        super().reset(seed=seed)
        self._seeded_once = True

        idx = self.np_random.integers(0, len(self.sessions))
        row = self.sessions.iloc[idx]

        self._state = {
            "soc": float(row["initial_soc"]),
            "target_soc": float(row["final_soc"]),
            "battery_kwh": float(row["battery_capacity_kWh"]),
            "power_limit": float(row["charging_power_kW"]),
            "price": float(row["electricity_price"]),
        }
        self._initial_hour = int(row["hour"])
        self._current_hour = int(row["hour"])
        self._n_steps = int(row["n_steps"])
        self._steps_remaining = self._n_steps
        self._step_count = 0

        return self._get_obs(), {"session_index": int(idx)}

    def step(self, action):
        # 1. Action translation: map [0, 1] action fraction to actual power (kW)
        action_fraction = float(np.clip(action[0], 0.0, 1.0))
        power = action_fraction * self._state["power_limit"]
        energy_kwh = power * self.dt_hours

        old_soc = self._state["soc"]
        new_soc = min(
            100.0,
            old_soc + (energy_kwh / self._state["battery_kwh"]) * 100.0,
        )
        self._state["soc"] = new_soc

        demand = self.hourly_demand[self._current_hour % 24]

        # 2. Useful SOC progress reward (clipped to avoid rewarding overcharging)
        eff_old = min(old_soc, self._state["target_soc"])
        eff_new = min(new_soc, self._state["target_soc"])
        soc_progress = max(0.0, (eff_new - eff_old) / 100.0)
        progress_reward = self.w_progress * soc_progress

        # 3. Normalized electricity cost penalty
        price_norm = np.clip(self._state["price"] / 15.0, 0.0, 1.5)
        energy_fraction = energy_kwh / self._state["battery_kwh"]
        cost_penalty = self.w_cost * price_norm * energy_fraction

        # 4. Quadratic grid peak penalty (rewards smoothing power consumption)
        demand_norm = np.clip(demand / 100.0, 0.0, 1.5)
        peak_penalty = self.w_peak * demand_norm * (action_fraction**2)

        reward = progress_reward - cost_penalty - peak_penalty

        # 5. Environment step update
        self._step_count += 1
        self._steps_remaining -= 1
        elapsed_hours = (self._step_count * self.step_minutes) // 60
        self._current_hour = (self._initial_hour + elapsed_hours) % 24

        terminated = self._steps_remaining <= 0
        truncated = False

        # 6. Terminal satisfaction evaluation
        if terminated:
            shortfall = max(0.0, self._state["target_soc"] - self._state["soc"])
            if shortfall <= 0.5:
                reward += self.w_terminal_bonus
            else:
                reward -= self.w_shortfall * (shortfall / 100.0)

        info = {
            "soc": self._state["soc"],
            "target_soc": self._state["target_soc"],
            "demand": demand,
            "energy_kwh": energy_kwh,
            "action_fraction": action_fraction,
            "soc_progress": soc_progress,
            "progress_reward": progress_reward,
            "cost_term": -cost_penalty,
            "peak_term": -peak_penalty,
        }

        return self._get_obs(), reward, terminated, truncated, info