"""
envs/battery_thermal_env.py

Core battery thermal management environment with time-varying heat generation profiles.
Designed for RL control experiments where the agent commands normalized cooling effort.

State idea:
    Observation = [T_batt, T_amb, q_gen, time_fraction, previous_action]

Thermal model:
    C_th * dT/dt = Q_gen(t) - h(u) * A * (T - T_amb) - P_cooling_direct(u)

Where:
    - Q_gen(t) is time-varying battery heat generation [W]
    - h(u) is airflow/liquid cooling heat transfer coefficient [W/m^2-K]
    - u is normalized cooling command in [0, 1]
    - P_cooling_direct is optional extra active cooling removal [W]

This file is intentionally self-contained so Phase 1 can run without the rest of the project.
"""


from __future__ import annotations
from typing import Tuple, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "gymnasium is required. Install it with: pip install gymnasium"
    ) from exc

HeatProfileFn = Callable[[float, np.random.Generator], float]
AmbientProfileFn = Callable[[float, np.random.Generator], float]

@dataclass
class BatteryThermalConfig:
    """Configuration parameters for the battery thermal simulator."""
    #Simulation timing
    total_time: float = 600.0          # [s]
    dt: float = 1.0                    # [s]

    # Thermal plant parameters
    initial_temp: float = 25.0         # [degC]
    ambient_temp: float = 25.0         # [degC]
    thermal_capacitance: float = 50_000.0  # [J/K]
    surface_area: float = 1.0          # [m^2]

    # Convective cooling model
    h_min: float = 5.0                 # [W/m^2-K]
    h_max: float = 80.0                # [W/m^2-K]
    cooling_nonlinearity: float = 1.0  # h(u)=h_min+(h_max-h_min)*u^gamma

    # Optional direct active cooling term, useful for compressor/chiller abstraction
    direct_cooling_max: float = 0.0    # [W]

    # Battery operating limits
    target_temp: float = 35.0          # [degC]
    soft_max_temp: float = 45.0        # [degC]
    hard_max_temp: float = 60.0        # [degC]
    min_temp: float = 0.0              # [degC]

    # Default heat generation profile settings
    base_heat: float = 500.0           # [W]
    heat_amplitude: float = 250.0      # [W]
    heat_period: float = 180.0         # [s]
    heat_noise_std: float = 15.0       # [W]
    heat_spike_probability: float = 0.01
    heat_spike_min: float = 300.0      # [W]
    heat_spike_max: float = 900.0      # [W]

    # Ambient profile settings
    ambient_amplitude: float = 2.0     # [degC]
    ambient_period: float = 900.0      # [s]
    ambient_noise_std: float = 0.05    # [degC]

    # Reward weights
    temp_error_weight: float = 1.0
    over_temp_weight: float = 8.0
    action_weight: float = 0.03
    action_smoothness_weight: float = 0.08
    hard_violation_penalty: float = 100.0

    # Observation normalization constants
    temp_obs_low: float = -10.0
    temp_obs_high: float = 80.0
    heat_obs_high: float = 2_000.0

    # Reproducibility
    seed: Optional[int] = None

class BatteryThermalEnv(gym.Env):
    """
    Gymnasium environment for battery thermal management.

    Action:
        Box([0], [1])
        0 = minimum cooling
        1 = maximum cooling

    Observation:
        Box shape (5,):
            [battery_temp_C,
             ambient_temp_C,
             heat_generation_W,
             time_fraction,
             previous_action]

    The environment is deliberately simple enough for Phase 1, but not so dumb that
    an RL agent learns a meaningless policy. Time-varying Q_gen matters.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        config: Optional[BatteryThermalConfig] = None,
        heat_profile: Optional[HeatProfileFn] = None,
        ambient_profile: Optional[AmbientProfileFn] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.config = config or BatteryThermalConfig()
        self.heat_profile = heat_profile or self.default_heat_profile
        self.ambient_profile = ambient_profile or self.default_ambient_profile
        self.render_mode = render_mode

        self.rng = np.random.default_rng(self.config.seed)

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    self.config.temp_obs_low,
                    self.config.temp_obs_low,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.config.temp_obs_high,
                    self.config.temp_obs_high,
                    self.config.heat_obs_high,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.time: float = 0.0
        self.step_count: int = 0
        self.temperature: float = self.config.initial_temp
        self.ambient_temperature: float = self.config.ambient_temp
        self.heat_generation: float = self.config.base_heat
        self.previous_action: float = 0.0
        self.episode_log: Dict[str, list] = {}

    # -------------------------------------------------------------------------
    # Standard Gymnasium API
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        options = options or {}

        initial_temp = float(options.get("initial_temp", self.config.initial_temp))
        initial_ambient = float(options.get("ambient_temp", self.config.ambient_temp))

        # Small randomized starts help avoid a brittle controller/RL policy.
        randomize = bool(options.get("randomize", True))
        if randomize:
            initial_temp += float(self.rng.normal(0.0, 0.75))
            initial_ambient += float(self.rng.normal(0.0, 0.25))

        self.time = 0.0
        self.step_count = 0
        self.temperature = initial_temp
        self.ambient_temperature = initial_ambient
        self.previous_action = 0.0
        self.heat_generation = float(self.heat_profile(self.time, self.rng))

        self.episode_log = {
            "time": [],
            "temperature": [],
            "ambient_temperature": [],
            "heat_generation": [],
            "action": [],
            "h_coeff": [],
            "cooling_power": [],
            "reward": [],
        }

        observation = self._get_observation()
        info = self._get_info(action=0.0, h_coeff=self.config.h_min, cooling_power=0.0)
        return observation, info

    def step(self, action: np.ndarray | float) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the thermal simulation by one time step."""
        u = self._sanitize_action(action)

        self.ambient_temperature = float(self.ambient_profile(self.time, self.rng))
        self.heat_generation = float(self.heat_profile(self.time, self.rng))

        h_coeff = self._cooling_coefficient(u)
        convective_cooling = h_coeff * self.config.surface_area * (
            self.temperature - self.ambient_temperature
        )
        direct_cooling = self.config.direct_cooling_max * u
        cooling_power = convective_cooling + direct_cooling

        dTdt = (self.heat_generation - cooling_power) / self.config.thermal_capacitance
        self.temperature += dTdt * self.config.dt

        reward = self._compute_reward(u)

        self.time += self.config.dt
        self.step_count += 1

        terminated = bool(
            self.temperature >= self.config.hard_max_temp
            or self.temperature <= self.config.min_temp
        )
        truncated = bool(self.time >= self.config.total_time)

        info = self._get_info(action=u, h_coeff=h_coeff, cooling_power=cooling_power)
        self._log_step(u, h_coeff, cooling_power, reward)

        self.previous_action = u

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print a compact human-readable state line."""
        print(
            f"t={self.time:7.1f}s | "
            f"T={self.temperature:6.2f}C | "
            f"Tamb={self.ambient_temperature:6.2f}C | "
            f"Q={self.heat_generation:7.2f}W | "
            f"u={self.previous_action:4.2f}"
        )

    def close(self) -> None:
        """Nothing special to close for this environment."""
        pass

    # -------------------------------------------------------------------------
    # Profiles
    # -------------------------------------------------------------------------

    def default_heat_profile(self, t: float, rng: np.random.Generator) -> float:
        """
        Default time-varying heat generation profile.

        Combines:
            - base load
            - sinusoidal load variation
            - stochastic noise
            - occasional high-power spikes

        This is closer to a drive-cycle/load-cycle abstraction than constant heat.
        """
        cfg = self.config

        sinusoidal = cfg.heat_amplitude * np.sin(2.0 * np.pi * t / cfg.heat_period)
        noise = rng.normal(0.0, cfg.heat_noise_std)

        spike = 0.0
        if rng.random() < cfg.heat_spike_probability:
            spike = rng.uniform(cfg.heat_spike_min, cfg.heat_spike_max)

        q_gen = cfg.base_heat + sinusoidal + noise + spike
        return float(np.clip(q_gen, 0.0, cfg.heat_obs_high))

    def default_ambient_profile(self, t: float, rng: np.random.Generator) -> float:
        """Default slowly varying ambient temperature profile."""
        cfg = self.config

        ambient = (
            cfg.ambient_temp
            + cfg.ambient_amplitude * np.sin(2.0 * np.pi * t / cfg.ambient_period)
            + rng.normal(0.0, cfg.ambient_noise_std)
        )
        return float(ambient)

    @staticmethod
    def make_step_heat_profile(
        low_heat: float = 300.0,
        high_heat: float = 900.0,
        switch_time: float = 200.0,
        noise_std: float = 10.0,
    ) -> HeatProfileFn:
        """Create a simple low-to-high step heat profile."""

        def profile(t: float, rng: np.random.Generator) -> float:
            base = low_heat if t < switch_time else high_heat
            return float(max(0.0, base + rng.normal(0.0, noise_std)))

        return profile

    @staticmethod
    def make_pulse_heat_profile(
        base_heat: float = 350.0,
        pulse_heat: float = 1_100.0,
        pulse_start: float = 150.0,
        pulse_end: float = 260.0,
        noise_std: float = 10.0,
    ) -> HeatProfileFn:
        """Create a finite-duration high-load pulse heat profile."""

        def profile(t: float, rng: np.random.Generator) -> float:
            base = pulse_heat if pulse_start <= t <= pulse_end else base_heat
            return float(max(0.0, base + rng.normal(0.0, noise_std)))

        return profile

    @staticmethod
    def make_drive_cycle_profile(
        points: np.ndarray,
        noise_std: float = 5.0,
    ) -> HeatProfileFn:
        """
        Create an interpolated drive-cycle heat profile.

        Parameters
        ----------
        points:
            Array with shape (N, 2), where column 0 is time [s] and column 1 is heat [W].
        noise_std:
            Standard deviation of heat noise [W].
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2): [time_s, heat_W]")

        times = points[:, 0]
        heats = points[:, 1]

        if np.any(np.diff(times) <= 0):
            raise ValueError("profile times must be strictly increasing")

        def profile(t: float, rng: np.random.Generator) -> float:
            q = np.interp(t, times, heats, left=heats[0], right=heats[-1])
            return float(max(0.0, q + rng.normal(0.0, noise_std)))

        return profile

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _sanitize_action(self, action: np.ndarray | float) -> float:
        """Convert user/agent action to scalar cooling command in [0, 1]."""
        if isinstance(action, np.ndarray):
            value = float(np.asarray(action).reshape(-1)[0])
        else:
            value = float(action)

        if not np.isfinite(value):
            value = 0.0

        return float(np.clip(value, 0.0, 1.0))

    def _cooling_coefficient(self, u: float) -> float:
        """Map normalized action to heat transfer coefficient."""
        cfg = self.config
        shaped_u = u ** cfg.cooling_nonlinearity
        return float(cfg.h_min + (cfg.h_max - cfg.h_min) * shaped_u)

    def _compute_reward(self, u: float) -> float:
        """
        Reward balances temperature tracking, overheating avoidance, energy use,
        and command smoothness.

        Higher is better. The reward is negative cost.
        """
        cfg = self.config

        temp_error = (self.temperature - cfg.target_temp) / max(1e-6, cfg.soft_max_temp - cfg.target_temp)
        temp_cost = cfg.temp_error_weight * temp_error**2

        over_temp = max(0.0, self.temperature - cfg.soft_max_temp)
        over_temp_cost = cfg.over_temp_weight * (over_temp / max(1e-6, cfg.hard_max_temp - cfg.soft_max_temp)) ** 2

        action_cost = cfg.action_weight * u**2
        smoothness_cost = cfg.action_smoothness_weight * (u - self.previous_action) ** 2

        hard_penalty = 0.0
        if self.temperature >= cfg.hard_max_temp or self.temperature <= cfg.min_temp:
            hard_penalty = cfg.hard_violation_penalty

        total_cost = temp_cost + over_temp_cost + action_cost + smoothness_cost + hard_penalty
        return -float(total_cost)

    def _get_observation(self) -> np.ndarray:
        """Return observation vector."""
        time_fraction = self.time / max(1e-6, self.config.total_time)
        obs = np.array(
            [
                self.temperature,
                self.ambient_temperature,
                self.heat_generation,
                np.clip(time_fraction, 0.0, 1.0),
                self.previous_action,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self, action: float, h_coeff: float, cooling_power: float) -> Dict[str, float]:
        """Return diagnostic info for logging/evaluation."""
        return {
            "time": float(self.time),
            "temperature": float(self.temperature),
            "ambient_temperature": float(self.ambient_temperature),
            "heat_generation": float(self.heat_generation),
            "action": float(action),
            "h_coeff": float(h_coeff),
            "cooling_power": float(cooling_power),
            "target_temp": float(self.config.target_temp),
            "soft_max_temp": float(self.config.soft_max_temp),
            "hard_max_temp": float(self.config.hard_max_temp),
        }

    def _log_step(self, action: float, h_coeff: float, cooling_power: float, reward: float) -> None:
        """Append current step data to episode log."""
        self.episode_log["time"].append(float(self.time))
        self.episode_log["temperature"].append(float(self.temperature))
        self.episode_log["ambient_temperature"].append(float(self.ambient_temperature))
        self.episode_log["heat_generation"].append(float(self.heat_generation))
        self.episode_log["action"].append(float(action))
        self.episode_log["h_coeff"].append(float(h_coeff))
        self.episode_log["cooling_power"].append(float(cooling_power))
        self.episode_log["reward"].append(float(reward))

    def get_episode_log(self) -> Dict[str, np.ndarray]:
        """Return episode log as numpy arrays."""
        return {key: np.asarray(value) for key, value in self.episode_log.items()}


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    config = BatteryThermalConfig(
        total_time=300.0,
        dt=1.0,
        initial_temp=25.0,
        base_heat=450.0,
        heat_amplitude=200.0,
        heat_spike_probability=0.02,
        direct_cooling_max=50.0,
        seed=7,
    )

    env = BatteryThermalEnv(config=config, render_mode="human")
    obs, info = env.reset(seed=7)

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        # Baseline controller: more cooling as temperature rises above target.
        temp = obs[0]
        action = np.array([np.clip((temp - 30.0) / 15.0, 0.0, 1.0)], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(f"\nEpisode finished. Total reward: {total_reward:.3f}")
    print(f"Final temperature: {info['temperature']:.2f} C")
