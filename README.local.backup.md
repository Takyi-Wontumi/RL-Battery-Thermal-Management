# RL Battery Thermal Management Controller

## Objective
Develop a reinforcement learning controller that regulates battery temperature under variable heat-generation and ambient conditions while minimizing cooling energy.

## System Model
Lumped thermal battery model:
C_th dT/dt = Q_gen - h(u)A(T_b - T_amb)

## Controllers Compared
- Rule-based thermostat
- PID controller
- PPO reinforcement learning controller

## RL Formulation
Observation:
[T_b, T_amb, Q_gen, previous_u, temperature_error]

Action:
continuous cooling command from 0 to 1

Reward:
temperature tracking penalty + overheating penalty + cooling energy penalty + actuator smoothness penalty

## Evaluation Metrics
- Maximum battery temperature
- RMS temperature error
- Cooling energy use
- Time above safe temperature
- Monte Carlo robustness
