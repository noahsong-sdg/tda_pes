#!/usr/bin/env python3
"""
Comparison of TDA methods using modular visualization system.

This script demonstrates how the TDAVisualizer class can work with 
any persistence computation method by standardizing the output format.
"""

import numpy as np
import sys
import os

# Import our modular TDA system
from ph_utils import (
    load_pes_data, 
    compute_alpha_sublevel,
    compute_rips_sublevel_filtration,
    create_universal_tda_analysis,
    TDAVisualizer
)

def compare_tda_methods(filename):
    """Compare multiple TDA methods using the modular visualization system."""
    
    print(f"Comparing TDA methods on: {filename}")
    print("=" * 60)
    
    # Load data
    coordinates, rel_energies = load_pes_data(filename)
    base_name = os.path.basename(filename).replace('.dat', '').replace('.csv', '')
    
    print(f"Loaded {len(coordinates)} points in {coordinates.shape[1]}D")
    print(f"Energy range: {rel_energies.min():.6f} to {rel_energies.max():.6f} kcal/mol\n")
    
    # Method 1: Alpha complex
    print("METHOD 1: Alpha Complex Sublevel Filtration")
    print("-" * 50)
    try:
        alpha_result = compute_alpha_sublevel(coordinates, rel_energies)
        
        # Use universal visualization system
        alpha_analysis = create_universal_tda_analysis(
            coordinates, rel_energies, 
            "Alpha Complex", alpha_result, "alpha", 
            f"{base_name}_alpha"
        )
        
        print("   ✓ Alpha complex analysis completed\n")
        
    except Exception as e:
        print(f"   ✗ Alpha complex failed: {e}\n")
        alpha_analysis = None
    
    # Method 2: Rips complex  
    print("METHOD 2: Rips Complex Sublevel Filtration")
    print("-" * 50)
    try:
        rips_result = compute_rips_sublevel_filtration(coordinates, rel_energies)
        
        # Use universal visualization system
        rips_analysis = create_universal_tda_analysis(
            coordinates, rel_energies,
            "Rips Complex", rips_result, "rips",
            f"{base_name}_rips"
        )
        
        print("   ✓ Rips complex analysis completed\n")
        
    except Exception as e:
        print(f"   ✗ Rips complex failed: {e}\n")
        rips_analysis = None
    
    # Method 3: Density-based (if we had it implemented)
    print("METHOD 3: Density-Based Sublevel Filtration")
    print("-" * 50)
    print("   ⚠ Not implemented yet, but would use same visualization system")
    print("   ⚠ Would call: create_universal_tda_analysis(..., 'density')\n")
    
    # Summary comparison
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if alpha_analysis and rips_analysis:
        # Extract feature counts for comparison
        alpha_features = alpha_analysis['standardized_result']['features']
        rips_features = rips_analysis['standardized_result']['features']
        
        print("Feature counts by method:")
        print(f"  Alpha Complex:")
        print(f"    H₀: {len(alpha_features['h0']['finite']) + len(alpha_features['h0']['essential'])}")
        print(f"    H₁: {len(alpha_features['h1']['finite']) + len(alpha_features['h1']['essential'])}")
        print(f"    H₂: {len(alpha_features['h2']['finite']) + len(alpha_features['h2']['essential'])}")
        
        print(f"  Rips Complex:")
        print(f"    H₀: {len(rips_features['h0']['finite']) + len(rips_features['h0']['essential'])}")
        print(f"    H₁: {len(rips_features['h1']['finite']) + len(rips_features['h1']['essential'])}")
        print(f"    H₂: {len(rips_features['h2']['finite']) + len(rips_features['h2']['essential'])}")
        
        print(f"\nVisualization files created:")
        print(f"  Alpha: {alpha_analysis['filename']}")
        print(f"  Rips:  {rips_analysis['filename']}")
    
    print("\nThe modular TDAVisualizer class ensures:")
    print("  ✓ Consistent visualization across all methods")
    print("  ✓ Standardized result format")
    print("  ✓ Color-coordinated features")
    print("  ✓ Spurious feature detection")
    print("  ✓ Easy addition of new TDA methods")

def demonstrate_modular_system():
    """Show how easy it is to add a new TDA method to the system."""
    
    print("\nDEMONSTRATING MODULAR SYSTEM")
    print("=" * 60)
    
    print("To add a new TDA method, you only need to:")
    print("1. Write a compute function that returns raw results")
    print("2. Add a case to TDAVisualizer.standardize_result()")
    print("3. Call create_universal_tda_analysis()")
    print()
    
    print("Example for a hypothetical 'watershed' method:")
    print("""
def compute_watershed_sublevel(coordinates, rel_energies):
    # Your watershed algorithm here
    return raw_watershed_result

# In TDAVisualizer.standardize_result():
elif method_type == 'watershed':
    # Convert watershed format to standard format
    return standardized_result

# Usage:
watershed_result = compute_watershed_sublevel(coords, energies)
analysis = create_universal_tda_analysis(
    coords, energies, "Watershed Method", 
    watershed_result, "watershed", "output_name"
)
""")
    
    print("The visualization will automatically work with:")
    print("  ✓ Same 2×2 layout (PES + Persistence + H₀/H₁ barcodes)")
    print("  ✓ Color coordination across all panels")
    print("  ✓ Spurious feature detection and marking")
    print("  ✓ Generator visualization on PES")
    print("  ✓ Consistent styling and formatting")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_methods.py <data_file>")
        print("Example: python compare_methods.py data/nh3_pes.dat")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found")
        sys.exit(1)
    
    # Run the comparison
    compare_tda_methods(filename)
    
    # Show how modular the system is
    demonstrate_modular_system() 
