#!/usr/bin/env python3
"""
Test script for proper sublevel filtration on molecular PES data.

This demonstrates the correct way to apply TDA to potential energy surfaces
using cubical complexes and sublevel filtration, as described in the GUDHI tutorial.

Examples:
    # Test on butane dihedral scan (periodic)
    python test_sublevel_filtration.py --molecule butane
    
    # Test on ammonia inversion (non-periodic)  
    python test_sublevel_filtration.py --molecule ammonia
    
    # Test on 2D conformational scan
    python test_sublevel_filtration.py --molecule 2d_example
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from ph_v2 import (
    compute_sublevel_cubical_persistence,
    compute_2d_sublevel_persistence, 
    compute_higher_dim_sublevel_persistence
)

try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    print("Warning: GUDHI not available")
    GUDHI_AVAILABLE = False


def create_butane_test_data():
    """Create test data resembling butane dihedral scan (periodic)."""
    angles = np.linspace(0, 360, 36, endpoint=False)  # Every 10 degrees
    
    # Butane energy profile: anti (180°) < gauche (±60°) << eclipsed (0°, 120°, 240°)
    energies = []
    for angle in angles:
        # Convert to principal value for energy calculation
        theta = np.radians(angle)
        
        # Anti conformation (global minimum around 180°)
        e_anti = 0.5 * (1 + np.cos(theta))  # Minimum at 180°
        
        # Gauche conformations (local minima around ±60°)
        e_gauche = 0.3 * (1 + np.cos(3*theta))  # Minima at 60°, 180°, 300°
        
        # Eclipsed conformations (maxima)
        e_eclipsed = 0.2 * (1 - np.cos(6*theta))  # Maxima at 0°, 60°, 120°, etc.
        
        # Combine contributions
        energy = e_anti + e_gauche + e_eclipsed
        energies.append(energy)
    
    energies = np.array(energies)
    # Convert to relative energies in Hartree (0.001 Hartree ≈ 0.63 kcal/mol)
    energies = (energies - np.min(energies)) * 0.005  # Scale to reasonable range
    
    return angles, energies


def create_ammonia_test_data():
    """Create test data resembling ammonia inversion (non-periodic)."""
    # Inversion coordinate from planar (0) to pyramidal (±1)
    coords = np.linspace(-1.2, 1.2, 25)
    
    # Double-well potential with barrier at planar geometry
    energies = []
    for x in coords:
        # Symmetric double well: minima at ±0.8, barrier at 0
        energy = 0.01 * (x**4 - 1.6*x**2 + 0.64)  # In Hartree
        energies.append(energy)
    
    energies = np.array(energies)
    energies = energies - np.min(energies)  # Make relative to minimum
    
    return coords, energies


def create_2d_test_data():
    """Create 2D conformational scan data (two dihedral angles)."""
    n_points = 20
    phi = np.linspace(0, 360, n_points, endpoint=False)
    psi = np.linspace(0, 360, n_points, endpoint=False)
    
    phi_grid, psi_grid = np.meshgrid(phi, psi)
    energy_grid = np.zeros((n_points, n_points))
    
    for i in range(n_points):
        for j in range(n_points):
            phi_rad = np.radians(phi_grid[i, j])
            psi_rad = np.radians(psi_grid[i, j])
            
            # Create a realistic 2D energy surface with multiple minima
            e1 = 0.3 * (1 + np.cos(phi_rad - np.pi))  # Minimum at phi=180°
            e2 = 0.2 * (1 + np.cos(psi_rad - np.pi/3))  # Minimum at psi=60°
            e3 = 0.1 * (1 + np.cos(2*phi_rad + psi_rad))  # Coupling term
            
            energy_grid[i, j] = e1 + e2 + e3
    
    # Make relative and convert to Hartree
    energy_grid = (energy_grid - np.min(energy_grid)) * 0.005
    
    return phi, psi, energy_grid


def test_1d_periodic():
    """Test 1D periodic data (butane dihedral scan)."""
    print("="*60)
    print("TEST 1: 1D Periodic Data (Butane-like dihedral scan)")
    print("="*60)
    
    angles, energies = create_butane_test_data()
    
    print(f"Generated {len(angles)} points from 0° to 360°")
    print(f"Energy range: {np.min(energies):.6f} to {np.max(energies):.6f} Hartree")
    print(f"Expected: Periodic data with transitions between conformers")
    
    # Apply sublevel filtration
    if GUDHI_AVAILABLE:
        persistence = compute_sublevel_cubical_persistence(angles, energies)
        
        h0_count = sum(1 for dim, _ in persistence if dim == 0)
        h1_count = sum(1 for dim, _ in persistence if dim == 1)
        
        print(f"\nResults: H0={h0_count}, H1={h1_count}")
        print("Expected: H1 > 0 for periodic transitions between conformational states")
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(angles, energies * 627.509, 'b-o', markersize=4)
        plt.xlabel('Dihedral Angle (degrees)')
        plt.ylabel('Relative Energy (kcal/mol)')
        plt.title('Test 1: Butane-like Dihedral Scan (Periodic)')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_1d_periodic.png', dpi=150, bbox_inches='tight')
        print("Saved plot: test_1d_periodic.png")
    else:
        print("GUDHI not available - skipping computation")


def test_1d_nonperiodic():
    """Test 1D non-periodic data (ammonia inversion)."""
    print("\n" + "="*60)
    print("TEST 2: 1D Non-Periodic Data (Ammonia-like inversion)")
    print("="*60)
    
    coords, energies = create_ammonia_test_data()
    
    print(f"Generated {len(coords)} points from -1.2 to +1.2 (inversion coordinate)")
    print(f"Energy range: {np.min(energies):.6f} to {np.max(energies):.6f} Hartree")
    print(f"Expected: Non-periodic double-well potential")
    
    # Apply sublevel filtration (treating as angles for compatibility)
    if GUDHI_AVAILABLE:
        # Scale coordinates to 0-360 range for existing function interface
        scaled_coords = (coords + 1.2) * 150  # Map [-1.2, 1.2] -> [0, 360]
        persistence = compute_sublevel_cubical_persistence(scaled_coords, energies)
        
        h0_count = sum(1 for dim, _ in persistence if dim == 0)
        h1_count = sum(1 for dim, _ in persistence if dim == 1)
        
        print(f"\nResults: H0={h0_count}, H1={h1_count}")
        print("Expected: H0 features for separate wells, minimal H1 (non-periodic)")
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(coords, energies * 627.509, 'r-o', markersize=4)
        plt.xlabel('Inversion Coordinate')
        plt.ylabel('Relative Energy (kcal/mol)')
        plt.title('Test 2: Ammonia-like Inversion (Non-Periodic)')
        plt.grid(True, alpha=0.3)
        plt.savefig('test_1d_nonperiodic.png', dpi=150, bbox_inches='tight')
        print("Saved plot: test_1d_nonperiodic.png")
    else:
        print("GUDHI not available - skipping computation")


def test_2d_conformational():
    """Test 2D conformational scan data."""
    print("\n" + "="*60)
    print("TEST 3: 2D Conformational Scan (Two dihedral angles)")
    print("="*60)
    
    phi, psi, energy_grid = create_2d_test_data()
    
    print(f"Generated {energy_grid.shape[0]}x{energy_grid.shape[1]} grid")
    print(f"Phi range: 0° to 360°, Psi range: 0° to 360°")
    print(f"Energy range: {np.min(energy_grid):.6f} to {np.max(energy_grid):.6f} Hartree")
    print(f"Expected: 2D energy landscape with multiple basins")
    
    # Apply 2D sublevel filtration
    if GUDHI_AVAILABLE:
        persistence = compute_2d_sublevel_persistence(energy_grid)
        
        h0_count = sum(1 for dim, _ in persistence if dim == 0)
        h1_count = sum(1 for dim, _ in persistence if dim == 1)
        h2_count = sum(1 for dim, _ in persistence if dim == 2)
        
        print(f"\nResults: H0={h0_count}, H1={h1_count}, H2={h2_count}")
        print("Expected: Multiple H0 (basins), H1 (transition paths), possible H2 (voids)")
        
        # Plot the 2D energy surface
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 2, 1)
        im = plt.contourf(phi, psi, energy_grid * 627.509, levels=20, cmap='viridis')
        plt.colorbar(im, label='Energy (kcal/mol)')
        plt.xlabel('Phi (degrees)')
        plt.ylabel('Psi (degrees)')
        plt.title('2D Energy Surface')
        
        plt.subplot(2, 2, 2)
        plt.contour(phi, psi, energy_grid * 627.509, levels=10, colors='black', alpha=0.6)
        plt.xlabel('Phi (degrees)')
        plt.ylabel('Psi (degrees)')
        plt.title('Energy Contours')
        
        plt.subplot(2, 2, 3)
        plt.imshow(energy_grid * 627.509, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Energy (kcal/mol)')
        plt.title('Energy Grid (for Cubical Complex)')
        plt.xlabel('Psi index')
        plt.ylabel('Phi index')
        
        plt.subplot(2, 2, 4)
        plt.hist(energy_grid.flatten() * 627.509, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Energy (kcal/mol)')
        plt.ylabel('Frequency')
        plt.title('Energy Distribution')
        
        plt.tight_layout()
        plt.savefig('test_2d_conformational.png', dpi=150, bbox_inches='tight')
        print("Saved plot: test_2d_conformational.png")
    else:
        print("GUDHI not available - skipping computation")


def main():
    """Run all tests demonstrating proper sublevel filtration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test proper sublevel filtration for molecular PES",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_sublevel_filtration.py --molecule butane
  python test_sublevel_filtration.py --molecule ammonia  
  python test_sublevel_filtration.py --molecule 2d
  python test_sublevel_filtration.py --all
        """
    )
    
    parser.add_argument('--molecule', choices=['butane', 'ammonia', '2d'], 
                       help='Test specific molecule type')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    print("SUBLEVEL FILTRATION TEST SUITE")
    print("Demonstrating proper cubical complex TDA for molecular PES")
    print("Based on GUDHI tutorial: https://github.com/GUDHI/TDA-tutorial")
    
    if not GUDHI_AVAILABLE:
        print("\nWARNING: GUDHI not available. Install with: pip install gudhi")
        print("Will generate data and plots but skip TDA computation.\n")
    
    if args.molecule == 'butane' or args.all:
        test_1d_periodic()
    
    if args.molecule == 'ammonia' or args.all:
        test_1d_nonperiodic()
    
    if args.molecule == '2d' or args.all:
        test_2d_conformational()
    
    if not args.molecule and not args.all:
        print("\nSpecify --molecule or --all to run tests")
        parser.print_help()
    
    print("\n" + "="*60)
    print("KEY ADVANTAGES OF SUBLEVEL FILTRATION:")
    print("="*60)
    print("✓ Uses natural grid structure of conformational space")
    print("✓ No artificial embeddings or arbitrary parameters")
    print("✓ Proper periodic boundary conditions when appropriate")
    print("✓ Direct interpretation: topology of {E ≤ threshold} sets")
    print("✓ Scales naturally to higher-dimensional conformational spaces")
    print("✓ Mathematically rigorous foundation in persistent homology")


if __name__ == "__main__":
    main() 
