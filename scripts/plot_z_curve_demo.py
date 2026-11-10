import torch
import matplotlib.pyplot as plt
import os

# Create 50 random points in 2D
torch.manual_seed(42)
points = torch.rand(100, 2)

# Scale them to a 16x16 grid (4-bit quantization for simple visual)
qmax = 15
points_q = torch.round(points * qmax).to(torch.int32)

# Compute Morton Code (Z-order)
morton_code = torch.zeros(points_q.shape[0], dtype=torch.int32)
for i in range(4): # 4 bits since qmax is 15
    bit_x = (points_q[:, 0] >> i) & 1
    bit_y = (points_q[:, 1] >> i) & 1
    morton_code |= (bit_x << (2 * i))
    morton_code |= (bit_y << (2 * i + 1))

# Sort points by Z-order
sorted_indices = torch.argsort(morton_code)
sorted_points = points_q[sorted_indices]

# Plot
plt.figure(figsize=(8, 8))
plt.title("Visualização Prática: Como a Curva-Z 'cata' os Gaussianos")

# Draw the 16x16 grid
plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
plt.xticks(range(16))
plt.yticks(range(16))

# Plot the path (the Z-curve sequence)
x_path = sorted_points[:, 0].numpy()
y_path = sorted_points[:, 1].numpy()
plt.plot(x_path, y_path, color='blue', alpha=0.5, linestyle='-', linewidth=2, label="Trajeto da Curva-Z")

# Plot the start and end
plt.scatter(x_path[0], y_path[0], color='green', s=100, label="Início (Primeiro na Sequência)", zorder=5)
plt.scatter(x_path[-1], y_path[-1], color='red', s=100, label="Fim (Último na Sequência)", zorder=5)

# Plot the points
plt.scatter(x_path, y_path, color='black', zorder=4)

plt.legend()
plt.tight_layout()
os.makedirs("results/debug", exist_ok=True)
plt.savefig("results/debug/z_curve_demo.png", dpi=150)
print("Saved to results/debug/z_curve_demo.png")
