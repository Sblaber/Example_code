import numpy as np
import matplotlib.pyplot as plt

# Define the energy landscape and its gradient
def energy_landscape(x, y):
    """Energy function U(x, y)"""
    return (x**2 - 1)**2 + (y**2 - 1)**2

def gradient_energy(x, y):
    """Gradient of U(x, y)"""
    dU_dx = 4 * x * (x**2 - 1)
    dU_dy = 4 * y * (y**2 - 1)
    return np.stack((dU_dx, dU_dy), axis=-1)  # Return a 2D array for all particles

# Simulation parameters
dt = 0.01  # Time step
gamma = 1.0  # Friction coefficient
kT = 0.1  # Thermal energy
n_steps = 10000  # Number of time steps
n_particles = 1  # Number of particles
x_limits = [-2.0, 2.0]  # x-axis bounds
y_limits = [-2.0, 2.0]  # y-axis bounds

# Initialize particle positions randomly within bounds
positions = np.random.uniform(
    low=[x_limits[0], y_limits[0]], 
    high=[x_limits[1], y_limits[1]], 
    size=(n_particles, 2)
)

# Store particle trajectories
trajectories = np.zeros((n_steps, n_particles, 2))
trajectories[0] = positions

# Function to apply periodic boundary conditions
def apply_periodic_boundary(positions, x_limits, y_limits):
    positions[:, 0] = (positions[:, 0] - x_limits[0]) % (x_limits[1] - x_limits[0]) + x_limits[0]
    positions[:, 1] = (positions[:, 1] - y_limits[0]) % (y_limits[1] - y_limits[0]) + y_limits[0]
    return positions

# Simulation loop (vectorized update for all particles)
for t in range(1, n_steps):
    # Compute the deterministic drift term (force = -grad(U))
    drift = -gradient_energy(positions[:, 0], positions[:, 1]) / gamma

    # Compute the stochastic term
    noise = np.sqrt(2 * kT / gamma) * np.random.normal(size=positions.shape)

    # Update all positions using Euler-Maruyama
    positions += drift * dt + noise * np.sqrt(dt)

    # Apply periodic boundary conditions
    positions = apply_periodic_boundary(positions, x_limits, y_limits)

    # Store trajectories
    trajectories[t] = positions

# Plot the results
# Energy landscape
x = np.linspace(x_limits[0], x_limits[1], 100)
y = np.linspace(y_limits[0], y_limits[1], 100)
X, Y = np.meshgrid(x, y)
U = energy_landscape(X, Y)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, U, levels=50, cmap="viridis")
plt.colorbar(label="Energy")

# Plot particle trajectories
for i in range(n_particles):
    plt.plot(trajectories[:, i, 0], trajectories[:, i, 1], lw=1)

plt.scatter(trajectories[0, :, 0], trajectories[0, :, 1], color='red', label='Start', zorder=5)
plt.scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], color='blue', label='End', zorder=5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Overdamped Brownian Dynamics with Periodic Boundary Conditions")
plt.show()
