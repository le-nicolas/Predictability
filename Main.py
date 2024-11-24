import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#If you think about it, some part of it involves the systematic machine learning of today.
#but i just get some chunks of it and try to make the thing work :)

# Step 1: Simulate data
np.random.seed(42)  # For reproducibility
true_mass = 5.0  # True mass
acceleration = np.linspace(0, 10, 50)  # Simulated acceleration
true_force = true_mass * acceleration  # True force (F = ma)
noise = np.random.normal(0, 5, size=acceleration.shape)  # Add noise
observed_force = true_force + noise  # Observed force with noise

# Step 2: Define the loss function
def loss_function(m):
    predicted_force = m * acceleration
    return np.sum((observed_force - predicted_force) ** 2)

# Step 3: Optimize mass using the loss function
initial_guess = 1.0  # Starting guess for mass
result = minimize(loss_function, initial_guess)# similar to newtons square root method.
optimized_mass = result.x[0]

# Step 4: Predicted force using the optimized mass
optimized_force = optimized_mass * acceleration

# Step 5: Visualization
plt.figure(figsize=(10, 6))

# Plot noisy observed data
plt.scatter(acceleration, observed_force, label="Observed Force (Noisy)", color="blue", alpha=0.7)

# Plot true force
plt.plot(acceleration, true_force, label=f"True Force (m={true_mass})", color="green", linestyle="--")

# Plot optimized force
plt.plot(acceleration, optimized_force, label=f"Optimized Force (m={optimized_mass:.2f})", color="red")

# Add labels and legend
plt.xlabel("Acceleration (a)")
plt.ylabel("Force (F)")
plt.title("Optimization Perspective of F = ma")
plt.legend()
plt.grid()

# Show plot
plt.show()
