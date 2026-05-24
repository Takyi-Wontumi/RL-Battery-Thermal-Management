import numpy as np
import matplotlib.pyplot as plt

def simulate_battery(cooling_command, total_time=1000, dt=1.0, T_initial=25.0, T_amb=25.0, Q_gen=500.0, C_th=50000.0, A=1.0, h_min=5.0, h_max=80.0):
    """
    Simulate the battery thermal behavior given a cooling command.

    Parameters:
    - cooling_command: A function that takes time as input and returns the cooling power (W).
    - total_time: Total simulation time in seconds.
    - dt: Time step in seconds.
    - T_initial: Initial battery temperature in degrees Celsius.
    - T_amb: Ambient temperature in degrees Celsius.
    - Q_gen: Heat generation rate in watts.
    - C_th: Thermal capacitance of the battery in J/°C.
    - A: Surface area of the battery in m².
    - h_min: Minimum heat transfer coefficient in W/(m²·°C).
    - h_max: Maximum heat transfer coefficient in W/(m²·°C).

    Returns:
    - time_array: Array of time points.
    - temperature_array: Array of battery temperatures corresponding to each time point.
    """

    cooling_command = np.clip(cooling_command, 0.0, 1.0)
    time = np.arange(0, total_time + dt, dt)
    temperature = np.zeros_like(time, dtype=float)
    temperature[0] = T_initial

    h = h_min + (h_max - h_min) * cooling_command

    for idx in range(len(time) - 1):
        T_b = temperature[idx]

        heat_removed = h * A * (T_b - T_amb)
        dTdt = (Q_gen - heat_removed) / C_th

        temperature[idx + 1] = T_b + dTdt * dt

    return time, temperature

if __name__ == "__main__":
    cooling_cases = {
        "No Cooling, u=0.0": 0.0,
        "Moderate Cooling, u=0.5": 0.5,
        "Full Cooling, u=1.0": 1.0
    }

    plt.figure(figsize=(10, 6))
    for label, u in cooling_cases.items():
        time, temperature = simulate_battery(cooling_command=u)
        plt.plot(time, temperature, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title("Battery Temperature vs. Time")
    plt.legend()
    plt.grid(True)
    plt.show()