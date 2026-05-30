import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =====================================================
# 1. Simulation Parameters to boil water vs heat by energy balance model
# =====================================================
T0 = 25.0

kettle_radius = 0.1
A = np.pi * kettle_radius**2
cover = True

dt = 1.0
num_simulations = 20

h_conv = 10.0
eta_cover = 0.85 if cover else 0.5

fire_mean = 1500
fire_sigma = 200

water_categories = [1, 2, 3, 4, 5]  # liters
c_water = 4184  # J/kg°C

# Elevation (m)
elevation = 1500  # example: 1500 m above sea level

# =====================================================
# 2. Environment Noise Function
# =====================================================
def stochastic_fire():
    return np.random.normal(fire_mean, fire_sigma)

def stochastic_air_temp():
    """Random ambient temperature per simulation ±5°C"""
    return 25.0 + np.random.uniform(-5, 5)

def boiling_temp(elevation_m):
    """Approximate boiling temperature with elevation"""
    return 100 - 0.003 * elevation_m  # °C

# =====================================================
# 3. Simulation Function (Single Run)
# =====================================================
def simulate_boil(m_l, elevation_m):
    """Return temperature evolution over time for a single stochastic run"""
    m = m_l
    T = T0
    temps = [T0]
    times = [0]
    t = 0
    fire_power = stochastic_fire()
    T_air = stochastic_air_temp()
    Tb = boiling_temp(elevation_m)
    
    while T < Tb:
        Q_in = fire_power * dt * eta_cover
        Q_loss = h_conv * A * (T - T_air) * dt
        dQ = Q_in - Q_loss
        dT = dQ / (m * c_water)
        T += dT
        t += dt
        temps.append(T)
        times.append(t)
    return np.array(times), np.array(temps)

# =====================================================
# 4. Run Simulations for All Water Amounts
# =====================================================
results = {}
temps_over_time = {}

for m_l in water_categories:
    boil_times = []
    temp_curves = []
    for sim in range(num_simulations):
        times, temps = simulate_boil(m_l, elevation)
        boil_times.append(times[-1])
        temp_curves.append((times, temps))
    results[m_l] = np.array(boil_times)
    temps_over_time[m_l] = temp_curves

# =====================================================
# 5. Print Results
# =====================================================
print("=== Boil Time Simulation (s) per Water Amount ===")
for m_l in water_categories:
    times = results[m_l]
    print(f"{m_l} L: Mean={np.mean(times):.1f}s, Std={np.std(times):.1f}s")

# =====================================================
# 6. Plot and Save Histogram as JPEG
# =====================================================
plt.figure(figsize=(12,6))
for m_l in water_categories:
    plt.hist(results[m_l], bins=10, alpha=0.5, label=f"{m_l} L")

plt.xlabel("Boil Time (s)")
plt.ylabel("Frequency")
plt.title(f"Stochastic Boil Time (Elevation {elevation} m)")
plt.legend()
plt.grid(True)
plt.savefig("boil_time_histogram.jpeg", dpi=300)
plt.show()

# =====================================================
# 7. Animate Temperature Evolution (Video)
# =====================================================
fig, ax = plt.subplots(figsize=(10,6))
lines = {m_l: ax.plot([], [], label=f"{m_l} L")[0] for m_l in water_categories}

ax.set_xlim(0, max(max(t[-1] for t,_ in temp_curves) for temp_curves in temps_over_time.values()))
ax.set_ylim(T0, boiling_temp(elevation) + 5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Stochastic Boil Temperature (Elevation {elevation} m)")
ax.legend()
ax.grid(True)

max_len = max(len(temps_over_time[m_l][0][0]) for m_l in water_categories)

def animate(frame):
    for m_l in water_categories:
        times, temps = temps_over_time[m_l][0]  # first run for animation
        if frame < len(times):
            lines[m_l].set_data(times[:frame], temps[:frame])
        else:
            lines[m_l].set_data(times, temps)
    return lines.values()

ani = FuncAnimation(fig, animate, frames=max_len, interval=50, blit=False)
writer = FFMpegWriter(fps=20)
ani.save("boil_simulation.mp4", writer=writer)

plt.close(fig)
