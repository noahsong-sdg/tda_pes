#!/usr/bin/env python3
"""
Compare Alpha vs Rips complexes for 1D PES data
This shows why Alpha complex creates spurious H1 features in 1D
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import gudhi
from pathlib import Path

def load_pes_data(filename):
    """Load PES data from file with comment filtering."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                data.append([float(x) for x in line.split()])
    
    data = np.array(data)
    coordinates = data[:, :-1]
    if coordinates.shape[1] == 0:
        coordinates = np.arange(len(data)).reshape(-1, 1)
    energies = data[:, -1]
    rel_energies = energies - energies.min()
    
    return coordinates, rel_energies

def analyze_complex(complex_type, coordinates, rel_energies):
    """Analyze a specific complex type."""
    print(f"\n{'='*50}")
    print(f"{complex_type.upper()} COMPLEX ANALYSIS")
    print(f"{'='*50}")
    
    if complex_type == "alpha":
        alpha_complex = gudhi.AlphaComplex(points=coordinates.tolist())
        simplex_tree = alpha_complex.create_simplex_tree()
    else:  # rips
        from scipy.spatial.distance import pdist
        distances = pdist(coordinates)
        max_edge_length = np.percentile(distances, 75)
        rips_complex = gudhi.RipsComplex(points=coordinates.tolist(), max_edge_length=max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    print(f"Complex: {simplex_tree.num_vertices()} vertices, {simplex_tree.num_simplices()} simplices")
    
    # Assign energy values
    for i, energy in enumerate(rel_energies):
        simplex_tree.assign_filtration([i], energy)
    
    # Extend filtration
    simplex_tree.extend_filtration()
    persistence = simplex_tree.persistence()
    
    print("\nPersistence intervals:")
    sublevel_features = []
    superlevel_features = []
    essential_features = []
    
    for p in persistence:
        dim, (birth, death) = p[0], p[1]
        
        if death == float('inf'):
            essential_features.append((dim, birth, death))
            print(f"  H{dim}: [{birth:.6f}, ∞] (ESSENTIAL)")
        elif birth < 0 or death > rel_energies.max() * 1.1:
            superlevel_features.append((dim, birth, death))
            print(f"  H{dim}: [{birth:.6f}, {death:.6f}] (SUPERLEVEL - likely spurious)")
        else:
            sublevel_features.append((dim, birth, death))
            print(f"  H{dim}: [{birth:.6f}, {death:.6f}] (sublevel)")
    
    return {
        'sublevel': sublevel_features,
        'superlevel': superlevel_features,
        'essential': essential_features,
        'persistence': persistence
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_alpha_rips.py <datafile>")
        sys.exit(1)
    
    filename = sys.argv[1]
    coordinates, rel_energies = load_pes_data(filename)
    
    print(f"Analyzing: {filename}")
    print(f"Data: {len(coordinates)} points in {coordinates.shape[1]}D")
    print(f"Energy range: [{rel_energies.min():.6f}, {rel_energies.max():.6f}]")
    
    # Compare both methods
    alpha_result = analyze_complex("alpha", coordinates, rel_energies)
    rips_result = analyze_complex("rips", coordinates, rel_energies)
    
    print(f"\n{'='*50}")
    print("COMPARISON SUMMARY")
    print(f"{'='*50}")
    
    print(f"ALPHA complex:")
    print(f"  Sublevel features: {len(alpha_result['sublevel'])}")
    print(f"  Superlevel features: {len(alpha_result['superlevel'])} (spurious)")
    print(f"  Essential features: {len(alpha_result['essential'])}")
    
    print(f"\nRIPS complex:")
    print(f"  Sublevel features: {len(rips_result['sublevel'])}")
    print(f"  Superlevel features: {len(rips_result['superlevel'])} (spurious)")
    print(f"  Essential features: {len(rips_result['essential'])}")
    
    print(f"\nRECOMMENDATION:")
    if len(alpha_result['superlevel']) > len(rips_result['superlevel']):
        print("Use RIPS complex - it produces fewer spurious features")
    elif len(alpha_result['superlevel']) < len(rips_result['superlevel']):
        print("Use ALPHA complex - it produces fewer spurious features")
    else:
        print("Both methods show similar spurious feature counts")
    
    # For 1D data, recommend interpreting only essential features
    if coordinates.shape[1] == 1:
        print("\nFor 1D PES data, focus on:")
        print("- Essential H0 features (connected components)")
        print("- Short-lived sublevel features (local topology)")
        print("- IGNORE superlevel features (extended filtration artifacts)")

if __name__ == "__main__":
    main() 
