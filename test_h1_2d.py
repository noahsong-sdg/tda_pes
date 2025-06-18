#!/usr/bin/env python3
"""
Test H1 detection with synthetic 2D data that has a clear topological hole.
"""

import numpy as np
import matplotlib.pyplot as plt

# Create 2D synthetic data with a hole in the middle
print("Creating synthetic 2D data with topological hole...")

# Create a grid
n_points = 21
x = np.linspace(-2, 2, n_points)
y = np.linspace(-2, 2, n_points)
X, Y = np.meshgrid(x, y)

# Create function with a hole: high energy in center, low energy on ring
R = np.sqrt(X**2 + Y**2)

# Create a "donut" function: low energy ring around high energy center
# This should create H1 features when viewed as sublevel sets
Z = np.where(R < 0.5, 1.0,           # High energy in center (hole)
             np.where(R < 1.0, 0.1,  # Low energy ring  
                      0.5 + 0.3*R))   # Medium energy outside

# Flatten for data format
coords = np.column_stack([X.flatten(), Y.flatten()])
energies = Z.flatten()

# Save as test data
header = """# Synthetic 2D Test Data with Topological Hole
# Method: Donut function with central high-energy region
# Columns: X Y RelativeEnergy(Hartree)
# Should show H1 features representing the central hole"""

np.savetxt('data/test_donut_2d.dat', 
           np.column_stack([coords, energies]), 
           header=header, fmt='%12.6f %12.6f %12.6f')

print(f"Created test data: {coords.shape[0]} points")
print(f"Energy range: {np.min(energies):.6f} to {np.max(energies):.6f}")
print(f"Saved to data/test_donut_2d.dat")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Energy')
ax1.set_title('3D Donut Function\n(Should create H1 features)')

# 2D contour
contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('2D Contour Map\n(Central hole should give H1)')
plt.colorbar(contour, ax=ax2)

plt.tight_layout()
plt.savefig('figures/test_donut_function.png', dpi=300)
print("Saved visualization to figures/test_donut_function.png")

print("\nTest with TDA:")
print("python src/ph_st.py data/test_donut_2d.dat --output test_donut_h1")
print("\nExpected results:")
print("- H0: Multiple components that merge as energy threshold increases")
print("- H1: Should see H1 features representing the central hole!")
print("- The hole appears when we exclude the high-energy center from sublevel sets") 
