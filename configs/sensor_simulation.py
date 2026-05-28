"""
configs/sensor_simulation.py

Realistic sensor and actuator simulation layer for the 3D battery-pack thermal
environment.

Purpose
-------
This file prevents controllers from receiving perfect simulator states.

Instead of:
    true thermal/electrical state -> controller

Use:
    true thermal/electrical state -> SensorSimulation -> measured observation
                                  -> RL policy OR baseline controller

This should be used for both:
    1. RL policies: PPO/SAC observations
    2. Classical baselines: global PI/P/bang-bang and zone-wise PI/P/hysteresis

The goal is not to make the simulation complicated for its own sake. The goal is
to make the control problem closer to a real BMS implementation: sparse
temperature sensing, noise, bias, delay, voltage/current measurement uncertainty,
cooling actuator lag, and actuator feedback.

Recommended integration point:
    envs/battery_pack_thermal_env_3d.py

Typical control loop:
    true_state = env thermal/electrical state
    sensor_packet = sensor_sim.measure(...)
    obs = sensor_packet.to_rl_observation()
    action_cmd = controller(obs)
    action_applied = actuator_sim.step(action_cmd)
    env physics uses action_applied
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SensorConfig:
    """Sensor realism settings."""

    # Enable/disable full sensor realism layer.
    enabled: bool = True

    # Number of independent cooling/sensing zones.
    num_zones: int = 4

    # Temperature sensor placement.
    # If True: use only selected thermistor cells.
    # If False: zone measurements are computed from true zone max temperatures.
    use_sparse_thermistors: bool = True

    # Number of thermistors per zone. Real BMS setups rarely measure every cell.
    thermistors_per_zone: int = 2

    # Measurement noise.
    temp_noise_std_c: float = 0.20
    current_noise_std_a: float = 0.20
    pack_voltage_noise_std_v: float = 0.02
    group_voltage_noise_std_v: float = 0.005
    coolant_temp_noise_std_c: float = 0.15
    actuator_feedback_noise_std: float = 0.01

    # Sensor calibration bias. Bias is sampled at reset and held for the episode.
    temp_bias_range_c: float = 0.50
    current_bias_range_a: float = 0.10
    voltage_bias_range_v: float = 0.01

    # Sensor delay.
    sensor_delay_s: float = 2.0

    # Optional dropout. Use low values; dropout should be a stressor, not chaos.
    enable_sensor_dropout: bool = False
    dropout_probability: float = 0.002
    dropout_hold_last_value: bool = True

    # Optional low-pass filtering of measured temperatures.
    enable_lowpass_filter: bool = True
    lowpass_alpha: float = 0.35

    # Domain randomization for training.
    # If enabled, noise/bias/delay are randomized around the nominal values.
    enable_domain_randomization: bool = False
    temp_noise_scale_range: Tuple[float, float] = (0.5, 1.5)
    current_noise_scale_range: Tuple[float, float] = (0.5, 1.5)
    delay_scale_range: Tuple[float, float] = (0.5, 1.5)


@dataclass
class ActuatorConfig:
    """Cooling actuator realism settings."""

    enabled: bool = True
    num_zones: int = 4

    # Cooling command delay.
    cooling_delay_s: float = 5.0

    # Rate limit: maximum change in normalized cooling command per second.
    enable_rate_limit: bool = True
    max_cooling_rate_per_s: float = 0.05

    # Actuator effectiveness. 1.0 means command maps perfectly to actual cooling.
    # Set <1.0 to represent weak fans/pumps/valves.
    effectiveness: float = 1.0

    # Optional actuator fault.
    enable_actuator_fault: bool = False
    fault_zone: Optional[int] = None
    fault_stuck_value: Optional[float] = None  # e.g., 0.3 means stuck at 30%


# ---------------------------------------------------------------------------
# Output packet
# ---------------------------------------------------------------------------

@dataclass
class SensorPacket:
    """
    Controller-facing measurement packet.

    All controllers should use this packet or observations built from it.
    Baseline controllers should NOT use hidden true simulator states.
    """

    T_zone_meas_c: np.ndarray
    T_pack_max_est_c: float
    T_pack_mean_est_c: float
    T_gradient_est_c: float

    pack_current_meas_a: float
    pack_voltage_meas_v: float
    group_voltage_meas_v: np.ndarray
    soc_est: float

    coolant_inlet_temp_meas_c: float
    coolant_outlet_temp_meas_c: float

    u_actual_feedback: np.ndarray
    u_prev_command: np.ndarray

    # Useful diagnostics.
    thermistor_cell_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    sensor_fault_flags: Dict[str, bool] = field(default_factory=dict)

    def to_rl_observation(self) -> np.ndarray:
        """
        Build a compact RL observation vector.

        Recommended order:
            zone temperatures
            pack summary temperatures
            electrical measurements
            cooling/coolant measurements
            actuator feedback and previous command
        """
        obs = np.concatenate(
            [
                self.T_zone_meas_c.astype(np.float32),
                np.array(
                    [
                        self.T_pack_max_est_c,
                        self.T_pack_mean_est_c,
                        self.T_gradient_est_c,
                        self.pack_current_meas_a,
                        self.pack_voltage_meas_v,
                        self.soc_est,
                        self.coolant_inlet_temp_meas_c,
                        self.coolant_outlet_temp_meas_c,
                    ],
                    dtype=np.float32,
                ),
                self.group_voltage_meas_v.astype(np.float32),
                self.u_actual_feedback.astype(np.float32),
                self.u_prev_command.astype(np.float32),
            ]
        )
        return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# Sensor simulation
# ---------------------------------------------------------------------------

class SensorSimulation:
    """
    Simulates BMS-style measurements from true battery-pack states.

    This class should be reset at the beginning of each episode.
    """

    def __init__(
        self,
        cfg: SensorConfig,
        dt_s: float,
        zone_ids: np.ndarray,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.dt_s = float(dt_s)
        self.zone_ids = np.asarray(zone_ids, dtype=int)
        self.num_cells = int(len(zone_ids))
        self.num_zones = int(cfg.num_zones)
        self.rng = np.random.default_rng(seed)

        self.delay_steps = max(0, int(round(cfg.sensor_delay_s / self.dt_s)))
        self.temp_buffer: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)

        self.thermistor_cell_ids = self._choose_thermistor_cells()

        self.temp_bias = np.zeros(self.num_zones, dtype=np.float32)
        self.current_bias = 0.0
        self.voltage_bias = 0.0
        self.prev_T_zone_meas: Optional[np.ndarray] = None
        self.last_packet: Optional[SensorPacket] = None

        self._temp_noise_std = cfg.temp_noise_std_c
        self._current_noise_std = cfg.current_noise_std_a
        self._delay_steps_active = self.delay_steps

    def reset(self, T_cells_true_c: ArrayLike) -> None:
        T_cells = np.asarray(T_cells_true_c, dtype=np.float32)
        T_zone_true = self._zone_temperature_measurement_basis(T_cells)

        if self.cfg.enable_domain_randomization:
            temp_scale = self.rng.uniform(*self.cfg.temp_noise_scale_range)
            current_scale = self.rng.uniform(*self.cfg.current_noise_scale_range)
            delay_scale = self.rng.uniform(*self.cfg.delay_scale_range)

            self._temp_noise_std = self.cfg.temp_noise_std_c * temp_scale
            self._current_noise_std = self.cfg.current_noise_std_a * current_scale
            self._delay_steps_active = max(
                0,
                int(round(self.delay_steps * delay_scale)),
            )
            self.temp_buffer = deque(maxlen=self._delay_steps_active + 1)
        else:
            self._temp_noise_std = self.cfg.temp_noise_std_c
            self._current_noise_std = self.cfg.current_noise_std_a
            self._delay_steps_active = self.delay_steps
            self.temp_buffer = deque(maxlen=self._delay_steps_active + 1)

        self.temp_bias = self.rng.uniform(
            -self.cfg.temp_bias_range_c,
            self.cfg.temp_bias_range_c,
            size=self.num_zones,
        ).astype(np.float32)

        self.current_bias = float(
            self.rng.uniform(-self.cfg.current_bias_range_a, self.cfg.current_bias_range_a)
        )
        self.voltage_bias = float(
            self.rng.uniform(-self.cfg.voltage_bias_range_v, self.cfg.voltage_bias_range_v)
        )

        self.prev_T_zone_meas = None
        self.last_packet = None

        for _ in range(self._delay_steps_active + 1):
            self.temp_buffer.append(T_zone_true.copy())

    def measure(
        self,
        *,
        T_cells_true_c: ArrayLike,
        pack_current_true_a: float,
        pack_voltage_true_v: float,
        group_voltage_true_v: ArrayLike,
        soc_true: float,
        coolant_inlet_true_c: float,
        coolant_outlet_true_c: float,
        u_actual: ArrayLike,
        u_prev_command: ArrayLike,
    ) -> SensorPacket:
        """Convert true simulator states into noisy/delayed/sparse measurements."""
        if not self.cfg.enabled:
            return self._ideal_packet(
                T_cells_true_c=T_cells_true_c,
                pack_current_true_a=pack_current_true_a,
                pack_voltage_true_v=pack_voltage_true_v,
                group_voltage_true_v=group_voltage_true_v,
                soc_true=soc_true,
                coolant_inlet_true_c=coolant_inlet_true_c,
                coolant_outlet_true_c=coolant_outlet_true_c,
                u_actual=u_actual,
                u_prev_command=u_prev_command,
            )

        T_cells = np.asarray(T_cells_true_c, dtype=np.float32)
        T_zone_true_basis = self._zone_temperature_measurement_basis(T_cells)

        self.temp_buffer.append(T_zone_true_basis.copy())
        T_zone_delayed = self.temp_buffer[0]

        T_zone_meas = (
            T_zone_delayed
            + self.temp_bias
            + self.rng.normal(0.0, self._temp_noise_std, size=self.num_zones)
        ).astype(np.float32)

        sensor_fault_flags: Dict[str, bool] = {}

        if self.cfg.enable_sensor_dropout:
            dropout_mask = self.rng.random(self.num_zones) < self.cfg.dropout_probability
            if np.any(dropout_mask):
                sensor_fault_flags["temperature_dropout"] = True
                if self.cfg.dropout_hold_last_value and self.prev_T_zone_meas is not None:
                    T_zone_meas[dropout_mask] = self.prev_T_zone_meas[dropout_mask]
                else:
                    T_zone_meas[dropout_mask] = np.nan
            else:
                sensor_fault_flags["temperature_dropout"] = False

        # Replace NaN with last valid or conservative value.
        if np.any(np.isnan(T_zone_meas)):
            if self.prev_T_zone_meas is not None:
                T_zone_meas = np.where(np.isnan(T_zone_meas), self.prev_T_zone_meas, T_zone_meas)
            else:
                T_zone_meas = np.nan_to_num(T_zone_meas, nan=float(np.nanmax(T_cells)))

        if self.cfg.enable_lowpass_filter and self.prev_T_zone_meas is not None:
            alpha = float(self.cfg.lowpass_alpha)
            T_zone_meas = alpha * T_zone_meas + (1.0 - alpha) * self.prev_T_zone_meas

        self.prev_T_zone_meas = T_zone_meas.copy()

        pack_current_meas = float(
            pack_current_true_a
            + self.current_bias
            + self.rng.normal(0.0, self._current_noise_std)
        )

        pack_voltage_meas = float(
            pack_voltage_true_v
            + self.voltage_bias
            + self.rng.normal(0.0, self.cfg.pack_voltage_noise_std_v)
        )

        group_voltage_true = np.asarray(group_voltage_true_v, dtype=np.float32)
        group_voltage_meas = (
            group_voltage_true
            + self.rng.normal(0.0, self.cfg.group_voltage_noise_std_v, size=len(group_voltage_true))
        ).astype(np.float32)

        coolant_inlet_meas = float(
            coolant_inlet_true_c
            + self.rng.normal(0.0, self.cfg.coolant_temp_noise_std_c)
        )
        coolant_outlet_meas = float(
            coolant_outlet_true_c
            + self.rng.normal(0.0, self.cfg.coolant_temp_noise_std_c)
        )

        u_actual_arr = np.asarray(u_actual, dtype=np.float32)
        u_feedback = (
            u_actual_arr
            + self.rng.normal(0.0, self.cfg.actuator_feedback_noise_std, size=len(u_actual_arr))
        ).astype(np.float32)
        u_feedback = np.clip(u_feedback, 0.0, 1.0)

        packet = SensorPacket(
            T_zone_meas_c=T_zone_meas.astype(np.float32),
            T_pack_max_est_c=float(np.max(T_zone_meas)),
            T_pack_mean_est_c=float(np.mean(T_zone_meas)),
            T_gradient_est_c=float(np.max(T_zone_meas) - np.min(T_zone_meas)),
            pack_current_meas_a=pack_current_meas,
            pack_voltage_meas_v=pack_voltage_meas,
            group_voltage_meas_v=group_voltage_meas,
            soc_est=float(np.clip(soc_true, 0.0, 1.0)),
            coolant_inlet_temp_meas_c=coolant_inlet_meas,
            coolant_outlet_temp_meas_c=coolant_outlet_meas,
            u_actual_feedback=u_feedback,
            u_prev_command=np.asarray(u_prev_command, dtype=np.float32),
            thermistor_cell_ids=self.thermistor_cell_ids.copy(),
            sensor_fault_flags=sensor_fault_flags,
        )

        self.last_packet = packet
        return packet

    def _ideal_packet(self, **kwargs) -> SensorPacket:
        T_cells = np.asarray(kwargs["T_cells_true_c"], dtype=np.float32)
        T_zone = self._compute_zone_max(T_cells)

        return SensorPacket(
            T_zone_meas_c=T_zone.astype(np.float32),
            T_pack_max_est_c=float(np.max(T_cells)),
            T_pack_mean_est_c=float(np.mean(T_cells)),
            T_gradient_est_c=float(np.max(T_cells) - np.min(T_cells)),
            pack_current_meas_a=float(kwargs["pack_current_true_a"]),
            pack_voltage_meas_v=float(kwargs["pack_voltage_true_v"]),
            group_voltage_meas_v=np.asarray(kwargs["group_voltage_true_v"], dtype=np.float32),
            soc_est=float(np.clip(kwargs["soc_true"], 0.0, 1.0)),
            coolant_inlet_temp_meas_c=float(kwargs["coolant_inlet_true_c"]),
            coolant_outlet_temp_meas_c=float(kwargs["coolant_outlet_true_c"]),
            u_actual_feedback=np.asarray(kwargs["u_actual"], dtype=np.float32),
            u_prev_command=np.asarray(kwargs["u_prev_command"], dtype=np.float32),
            thermistor_cell_ids=self.thermistor_cell_ids.copy(),
            sensor_fault_flags={},
        )

    def _choose_thermistor_cells(self) -> np.ndarray:
        ids = []
        for z in range(self.num_zones):
            cells = np.where(self.zone_ids == z)[0]
            if len(cells) == 0:
                continue
            n = min(self.cfg.thermistors_per_zone, len(cells))
            chosen = self.rng.choice(cells, size=n, replace=False)
            ids.extend(chosen.tolist())
        return np.asarray(sorted(ids), dtype=int)

    def _zone_temperature_measurement_basis(self, T_cells: np.ndarray) -> np.ndarray:
        """
        Return zone-level temperature measurement basis.

        If sparse thermistors are enabled, each zone measurement is the max
        thermistor reading within that zone. Otherwise, it is the true zone max.
        """
        if not self.cfg.use_sparse_thermistors:
            return self._compute_zone_max(T_cells).astype(np.float32)

        T_zone = np.zeros(self.num_zones, dtype=np.float32)

        for z in range(self.num_zones):
            thermistor_ids_z = self.thermistor_cell_ids[
                self.zone_ids[self.thermistor_cell_ids] == z
            ]

            if len(thermistor_ids_z) > 0:
                T_zone[z] = float(np.max(T_cells[thermistor_ids_z]))
            else:
                # Fallback: use true zone max (should rarely happen)
                cells_z = np.where(self.zone_ids == z)[0]
                T_zone[z] = float(np.max(T_cells[cells_z]))

        return T_zone

    def _compute_zone_max(self, T_cells: np.ndarray) -> np.ndarray:
        T_zone = np.zeros(self.num_zones, dtype=np.float32)
        for z in range(self.num_zones):
            cells_z = np.where(self.zone_ids == z)[0]
            if len(cells_z) == 0:
                T_zone[z] = float(np.max(T_cells))
            else:
                T_zone[z] = float(np.max(T_cells[cells_z]))
        return T_zone


# ---------------------------------------------------------------------------
# Cooling actuator simulation
# ---------------------------------------------------------------------------

class CoolingActuatorSimulation:
    """
    Simulates cooling actuator delay, rate limits, saturation, and simple faults.

    Controller command:
        u_cmd in [0, 1]^num_zones

    Physics receives:
        u_actual in [0, 1]^num_zones
    """

    def __init__(
        self,
        cfg: ActuatorConfig,
        dt_s: float,
    ) -> None:
        self.cfg = cfg
        self.dt_s = float(dt_s)
        self.num_zones = int(cfg.num_zones)

        self.delay_steps = max(0, int(round(cfg.cooling_delay_s / self.dt_s)))
        self.buffer: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)

        self.u_prev_command = np.zeros(self.num_zones, dtype=np.float32)
        self.u_actual = np.zeros(self.num_zones, dtype=np.float32)

    def reset(self) -> None:
        self.u_prev_command = np.zeros(self.num_zones, dtype=np.float32)
        self.u_actual = np.zeros(self.num_zones, dtype=np.float32)
        self.buffer.clear()
        for _ in range(self.delay_steps + 1):
            self.buffer.append(np.zeros(self.num_zones, dtype=np.float32))

    def step(self, u_cmd: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            u_limited_command, u_actual_applied
        """
        u_cmd_arr = np.asarray(u_cmd, dtype=np.float32).reshape(-1)

        if len(u_cmd_arr) == 1 and self.num_zones > 1:
            u_cmd_arr = np.full(self.num_zones, float(u_cmd_arr[0]), dtype=np.float32)

        if len(u_cmd_arr) != self.num_zones:
            raise ValueError(
                f"Expected {self.num_zones} actuator commands, got {len(u_cmd_arr)}."
            )

        u_cmd_arr = np.clip(u_cmd_arr, 0.0, 1.0)

        if not self.cfg.enabled:
            self.u_prev_command = u_cmd_arr.copy()
            self.u_actual = u_cmd_arr.copy()
            return self.u_prev_command.copy(), self.u_actual.copy()

        # Rate limiting on commanded action.
        if self.cfg.enable_rate_limit:
            max_delta = self.cfg.max_cooling_rate_per_s * self.dt_s
            u_limited = np.clip(
                u_cmd_arr,
                self.u_prev_command - max_delta,
                self.u_prev_command + max_delta,
            )
        else:
            u_limited = u_cmd_arr.copy()

        u_limited = np.clip(u_limited, 0.0, 1.0)

        # Delay.
        self.buffer.append(u_limited.copy())
        u_actual = self.buffer[0].copy()

        # Effectiveness.
        u_actual = np.clip(self.cfg.effectiveness * u_actual, 0.0, 1.0)

        # Optional fault.
        if self.cfg.enable_actuator_fault:
            z = self.cfg.fault_zone
            if z is not None and 0 <= z < self.num_zones and self.cfg.fault_stuck_value is not None:
                u_actual[z] = float(np.clip(self.cfg.fault_stuck_value, 0.0, 1.0))

        self.u_prev_command = u_limited.copy()
        self.u_actual = u_actual.copy()

        return self.u_prev_command.copy(), self.u_actual.copy()


# ---------------------------------------------------------------------------
# Baseline-controller helpers
# ---------------------------------------------------------------------------

def build_global_baseline_observation(packet: SensorPacket) -> Dict:
    """Observation dictionary for old global controllers. Use instead of true simulator values."""
    return {
        "T_max": packet.T_pack_max_est_c,
        "T_mean": packet.T_pack_mean_est_c,
        "T_grad": packet.T_gradient_est_c,
        "current_A": packet.pack_current_meas_a,
        "SOC": packet.soc_est,
    }


def build_zone_baseline_observation(packet: SensorPacket) -> Dict:
    """Observation dictionary for new multi-zone classical controllers."""
    return {
        "T_zone": packet.T_zone_meas_c.copy(),
        "T_max": packet.T_pack_max_est_c,
        "T_mean": packet.T_pack_mean_est_c,
        "T_grad": packet.T_gradient_est_c,
        "current_A": packet.pack_current_meas_a,
        "SOC": packet.soc_est,
        "u_actual": packet.u_actual_feedback.copy(),
    }


def zonewise_proportional_controller(
    T_zone_c: ArrayLike,
    *,
    target_temp_c: float = 35.0,
    activation_temp_c: float = 32.0,
    kp_target: float = 0.15,
    kp_balance: float = 0.10,
) -> np.ndarray:
    """
    Simple zone-wise proportional controller.

    Cools zones that are above an activation threshold and/or hotter than pack mean.
    Minimum fair classical baseline for multi-zone RL.
    """
    T_zone = np.asarray(T_zone_c, dtype=np.float32)
    T_mean = float(np.mean(T_zone))

    u = (
        kp_target * np.maximum(0.0, T_zone - activation_temp_c)
        + kp_balance * np.maximum(0.0, T_zone - T_mean)
    )
    u += kp_target * np.maximum(0.0, T_zone - target_temp_c)

    return np.clip(u, 0.0, 1.0).astype(np.float32)


class ZonewisePIController:
    """
    Zone-wise PI baseline.

    This is the controller SAC/PPO must beat if you want to claim RL adds value
    beyond classical multi-zone feedback.
    """

    def __init__(
        self,
        num_zones: int,
        target_temp_c: float = 35.0,
        activation_temp_c: float = 32.0,
        kp: float = 0.14,
        ki: float = 0.002,
        k_balance: float = 0.08,
        integral_limit: float = 50.0,
    ) -> None:
        self.num_zones = int(num_zones)
        self.target_temp_c = float(target_temp_c)
        self.activation_temp_c = float(activation_temp_c)
        self.kp = float(kp)
        self.ki = float(ki)
        self.k_balance = float(k_balance)
        self.integral_limit = float(integral_limit)
        self.integral = np.zeros(self.num_zones, dtype=np.float32)

    def reset(self) -> None:
        self.integral[:] = 0.0

    def __call__(self, T_zone_c: ArrayLike, dt_s: float) -> np.ndarray:
        T_zone = np.asarray(T_zone_c, dtype=np.float32)
        T_mean = float(np.mean(T_zone))

        error = np.maximum(0.0, T_zone - self.activation_temp_c)
        self.integral += error * float(dt_s)
        self.integral = np.clip(self.integral, 0.0, self.integral_limit)

        balance_error = np.maximum(0.0, T_zone - T_mean)

        u = (
            self.kp * error
            + self.ki * self.integral
            + self.k_balance * balance_error
        )

        return np.clip(u, 0.0, 1.0).astype(np.float32)
