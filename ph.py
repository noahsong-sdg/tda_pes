#!/usr/bin/env python3
"""
Topological Data Analysis for Molecular Potential Energy Surfaces
Usage: python ph.py <datafile>

Produces comprehensive visualization and analysis for 1D, 2D, or 3D PES data.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import gudhi
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

def load_pes_data(filename):
    """Load PES data from file with comment filtering."""
    try:
        # Read data, skipping comment lines
        data = []
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    data.append([float(x) for x in line.split()])
        
        data = np.array(data)
        
        # Separate coordinates and energies
        coordinates = data[:, :-1]
        if coordinates.shape[1] == 0:  # Single column data (energies only)
            coordinates = np.arange(len(data)).reshape(-1, 1)
        energies = data[:, -1]
        
        # Convert to relative energies
        rel_energies = energies - energies.min()
        
        return coordinates, rel_energies
        
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None

def compute_rips_sublevel(coordinates, rel_energies):
    """Compute sublevel filtration using Rips complex (better for 1D data)."""
    print(f"Computing Rips sublevel filtration for {coordinates.shape[1]}D data...")
    
    # Estimate appropriate max edge length from coordinate distances
    from scipy.spatial.distance import pdist
    distances = pdist(coordinates)
    max_edge_length = np.percentile(distances, 85)
    
    print(f"   Using max edge length: {max_edge_length:.6f}")
    
    rips_complex = gudhi.RipsComplex(points=coordinates.tolist(), max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # Debug: Show some information about the complex
    print(f"   Complex: {simplex_tree.num_vertices()} vertices, {simplex_tree.num_simplices()} simplices")
    
    # Assign energy function values to vertices
    for i, energy in enumerate(rel_energies):
        simplex_tree.assign_filtration([i], energy)
    
    # Extend to sublevel filtration
    simplex_tree.extend_filtration()
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Debug: Show the actual persistence intervals
    print("   Debug - Raw persistence intervals:")
    for p in persistence:
        if p[1][1] != float('inf'):
            print(f"     H{p[0]}: [{p[1][0]:.6f}, {p[1][1]:.6f}] (length: {p[1][1] - p[1][0]:.6f})")
        else:
            print(f"     H{p[0]}: [{p[1][0]:.6f}, ∞]")
    
    # Extract features by dimension
    features = {}
    for dim in range(3):  # H0, H1, H2
        finite = [(p[1][0], p[1][1]) for p in persistence if p[0] == dim and p[1][1] != float('inf')]
        essential = [(p[1][0], float('inf')) for p in persistence if p[0] == dim and p[1][1] == float('inf')]
        features[f'h{dim}'] = {'finite': finite, 'essential': essential}
    
    print(f"   ✓ H0: {len(features['h0']['finite']) + len(features['h0']['essential'])} features")
    print(f"   ✓ H1: {len(features['h1']['finite']) + len(features['h1']['essential'])} features")
    print(f"   ✓ H2: {len(features['h2']['finite']) + len(features['h2']['essential'])} features")
    
    return {
        'persistence': persistence,
        'features': features,
        'simplex_tree': simplex_tree
    }

def compute_alpha_sublevel_main(coordinates, rel_energies):
    """Compute sublevel filtration using Alpha complex."""
    print(f"Computing Alpha sublevel filtration for {coordinates.shape[1]}D data...")
    
    # Alpha complex only works for dimensions <= 3
    if coordinates.shape[1] > 3:
        print(f"Warning: Alpha complex not recommended for {coordinates.shape[1]}D data")
        return None
    
    # For 1D data, warn about potential artifacts
    if coordinates.shape[1] == 1:
        print("   Warning: Alpha complex in 1D may create spurious H1 features")
        print("   Consider using Rips complex for 1D PES data")
    
    alpha_complex = gudhi.AlphaComplex(points=coordinates.tolist())
    simplex_tree = alpha_complex.create_simplex_tree()
    
    # Debug: Show some information about the complex
    print(f"   Complex: {simplex_tree.num_vertices()} vertices, {simplex_tree.num_simplices()} simplices")
    
    # Assign energy function values to vertices
    for i, energy in enumerate(rel_energies):
        simplex_tree.assign_filtration([i], energy)
    
    # Extend to sublevel filtration
    simplex_tree.extend_filtration()
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Debug: Show the actual persistence intervals
    print("   Debug - Raw persistence intervals:")
    for p in persistence:
        if p[1][1] != float('inf'):
            print(f"     H{p[0]}: [{p[1][0]:.6f}, {p[1][1]:.6f}] (length: {p[1][1] - p[1][0]:.6f})")
        else:
            print(f"     H{p[0]}: [{p[1][0]:.6f}, ∞]")
    
    # Extract features by dimension
    features = {}
    for dim in range(3):  # H0, H1, H2
        finite = [(p[1][0], p[1][1]) for p in persistence if p[0] == dim and p[1][1] != float('inf')]
        essential = [(p[1][0], float('inf')) for p in persistence if p[0] == dim and p[1][1] == float('inf')]
        features[f'h{dim}'] = {'finite': finite, 'essential': essential}
    
    print(f"   ✓ H0: {len(features['h0']['finite']) + len(features['h0']['essential'])} features")
    print(f"   ✓ H1: {len(features['h1']['finite']) + len(features['h1']['essential'])} features")
    print(f"   ✓ H2: {len(features['h2']['finite']) + len(features['h2']['essential'])} features")
    
    return {
        'persistence': persistence,
        'features': features,
        'simplex_tree': simplex_tree
    }

def extract_generators(simplex_tree, features):
    """Extract representative generators for persistence features using persistence pairs."""
    generators = {}
    
    try:
        # Use GUDHI's persistence pairs to get actual generators
        persistence_pairs = simplex_tree.persistence_pairs()
        
        # Initialize generators dictionary
        for dim in range(3):
            generators[f'h{dim}'] = []
        
        # Map persistence features to their generators
        for birth_simplex, death_simplex in persistence_pairs:
            dim = len(birth_simplex) - 1
            
            if dim <= 2:  # Only handle dimensions we care about
                # Get filtration values to match with persistence intervals
                birth_filtration = simplex_tree.filtration(birth_simplex)
                death_filtration = simplex_tree.filtration(death_simplex) if death_simplex else float('inf')
                
                generators[f'h{dim}'].append({
                    'birth': birth_filtration,
                    'death': death_filtration,
                    'birth_simplex': birth_simplex,
                    'death_simplex': death_simplex,
                    'dimension': dim
                })
                        
    except Exception as e:
        print(f"Warning: Could not extract generators from persistence pairs: {e}")
        # Fallback to simple feature extraction
        generators = {'h0': [], 'h1': [], 'h2': []}
        for dim in range(3):
            key = f'h{dim}'
            finite_features = features[key]['finite']
            for birth, death in finite_features:
                generators[key].append({
                    'birth': birth,
                    'death': death,
                    'birth_simplex': None,
                    'death_simplex': None,
                    'dimension': dim
                })
    
    return generators

def plot_pes_with_generators(coordinates, rel_energies, generators, title_prefix=""):
    """Create PES visualization with generators overlaid."""
    n_dims = coordinates.shape[1]
    
    # Define colors for different homology dimensions
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    if n_dims == 1:
        # 1D: Line plot with generator points highlighted
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot the PES curve
        ax.plot(coordinates.flatten(), rel_energies, 'k-', alpha=0.6, linewidth=1, label='PES')
        ax.scatter(coordinates.flatten(), rel_energies, c='lightgray', s=20, alpha=0.5)
        
        # Plot generators - show ALL generators but distinguish valid from spurious
        valid_count = {'h0': 0, 'h1': 0, 'h2': 0}
        spurious_count = {'h0': 0, 'h1': 0, 'h2': 0}
        
        for dim_key, gen_list in generators.items():
            if not gen_list:
                continue
                
            dim = int(dim_key[1])
            color = colors[dim % len(colors)]
            
            for i, gen in enumerate(gen_list):
                birth_simplex = gen.get('birth_simplex', None)
                death_simplex = gen.get('death_simplex', None)
                
                # Determine if spurious
                is_spurious = (gen['birth'] < -0.01 or 
                             (gen['death'] != float('inf') and gen['death'] > rel_energies.max() * 1.1))
                
                if is_spurious:
                    spurious_count[dim_key] += 1
                    alpha = 0.3
                    marker_style = 'x'
                    line_style = '--'
                    label_suffix = ' (spurious)'
                else:
                    valid_count[dim_key] += 1
                    alpha = 0.8
                    marker_style = 'o'
                    line_style = '-'
                    label_suffix = ''
                
                if birth_simplex is not None:
                    if dim == 0:  # H0: highlight birth vertex
                        vertex = birth_simplex[0]
                        if vertex < len(coordinates):
                            feature_num = valid_count[dim_key] if not is_spurious else spurious_count[dim_key]
                            ax.scatter(coordinates[vertex, 0], rel_energies[vertex], 
                                     c=color, s=100, marker=marker_style, alpha=alpha,
                                     label=f'H{dim} birth{label_suffix}' if (valid_count[dim_key] + spurious_count[dim_key]) == 1 else "")
                            ax.annotate(f'H{dim}.{feature_num}b{"*" if is_spurious else ""}', 
                                      (coordinates[vertex, 0], rel_energies[vertex]),
                                      xytext=(5, 5), textcoords='offset points', fontsize=8)
                    
                    elif dim == 1:  # H1: highlight edge
                        if len(birth_simplex) == 2:
                            v1, v2 = birth_simplex
                            if v1 < len(coordinates) and v2 < len(coordinates):
                                feature_num = valid_count[dim_key] if not is_spurious else spurious_count[dim_key]
                                # Draw line between vertices
                                ax.plot([coordinates[v1, 0], coordinates[v2, 0]], 
                                       [rel_energies[v1], rel_energies[v2]], 
                                       color=color, linewidth=3, alpha=alpha, linestyle=line_style,
                                       label=f'H{dim} edge{label_suffix}' if (valid_count[dim_key] + spurious_count[dim_key]) == 1 else "")
                                
                                # Annotate midpoint
                                mid_x = (coordinates[v1, 0] + coordinates[v2, 0]) / 2
                                mid_y = (rel_energies[v1] + rel_energies[v2]) / 2
                                ax.annotate(f'H{dim}.{feature_num}{"*" if is_spurious else ""}', 
                                          (mid_x, mid_y), xytext=(5, 5), 
                                          textcoords='offset points', fontsize=8)
                
                # Highlight death simplex if finite
                if death_simplex is not None and gen['death'] != float('inf'):
                    if dim == 0 and len(death_simplex) == 2:  # Edge that kills H0
                        v1, v2 = death_simplex
                        if v1 < len(coordinates) and v2 < len(coordinates):
                            feature_num = valid_count[dim_key] if not is_spurious else spurious_count[dim_key]
                            ax.plot([coordinates[v1, 0], coordinates[v2, 0]], 
                                   [rel_energies[v1], rel_energies[v2]], 
                                   color=color, linewidth=2, alpha=alpha*0.7, linestyle=':')
                            ax.annotate(f'H{dim}.{feature_num}d{"*" if is_spurious else ""}', 
                                      ((coordinates[v1, 0] + coordinates[v2, 0])/2, 
                                       (rel_energies[v1] + rel_energies[v2])/2),
                                      xytext=(5, -15), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Relative Energy (kcal/mol)')
        ax.set_title(f'{title_prefix}PES with Homology Generators (* = spurious)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    elif n_dims == 2:
        # 2D: 3D surface plot with generator highlighting
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the PES surface
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], rel_energies, 
                           c=rel_energies, cmap='viridis', s=20, alpha=0.6)
        
        # Plot generators
        feature_count = {'h0': 0, 'h1': 0, 'h2': 0}
        
        for dim_key, gen_list in generators.items():
            if not gen_list:
                continue
                
            dim = int(dim_key[1])
            color = colors[dim % len(colors)]
            
            for i, gen in enumerate(gen_list):
                # Skip spurious features
                if gen['birth'] < -0.01 or (gen['death'] != float('inf') and gen['death'] > rel_energies.max() * 1.1):
                    continue
                    
                feature_count[dim_key] += 1
                birth_simplex = gen.get('birth_simplex', None)
                
                if birth_simplex is not None:
                    if dim == 0:  # H0: highlight vertex
                        vertex = birth_simplex[0]
                        if vertex < len(coordinates):
                            ax.scatter(coordinates[vertex, 0], coordinates[vertex, 1], rel_energies[vertex], 
                                     c=color, s=200, marker='o', alpha=0.9)
                    
                    elif dim == 1:  # H1: highlight edge
                        if len(birth_simplex) == 2:
                            v1, v2 = birth_simplex
                            if v1 < len(coordinates) and v2 < len(coordinates):
                                ax.plot([coordinates[v1, 0], coordinates[v2, 0]], 
                                       [coordinates[v1, 1], coordinates[v2, 1]],
                                       [rel_energies[v1], rel_energies[v2]], 
                                       color=color, linewidth=4, alpha=0.9)
        
        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        ax.set_zlabel('Relative Energy (kcal/mol)')
        ax.set_title(f'{title_prefix}PES with Homology Generators')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Relative Energy (kcal/mol)')
        
    else:
        # Higher dimensions: project to first 2 dimensions
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           c=rel_energies, cmap='viridis', s=20, alpha=0.6)
        
        # Plot generators
        for dim_key, gen_list in generators.items():
            if not gen_list:
                continue
                
            dim = int(dim_key[1])
            color = colors[dim % len(colors)]
            
            for i, gen in enumerate(gen_list):
                # Skip spurious features
                if gen['birth'] < -0.01 or (gen['death'] != float('inf') and gen['death'] > rel_energies.max() * 1.1):
                    continue
                    
                birth_simplex = gen.get('birth_simplex', None)
                
                if birth_simplex is not None and dim == 0:  # Only show H0 for clarity
                    vertex = birth_simplex[0]
                    if vertex < len(coordinates):
                        ax.scatter(coordinates[vertex, 0], coordinates[vertex, 1], 
                                 c=color, s=100, marker='o', alpha=0.9)
        
        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        ax.set_title(f'{title_prefix}PES with Homology Generators ({n_dims}D data)')
        plt.colorbar(scatter, ax=ax, label='Relative Energy')
        
    plt.tight_layout()
    return fig

def create_combined_analysis(coordinates, rel_energies, generators, result, title_prefix=""):
    """Create a simplified TDA analysis with color-coordinated PES and persistence diagram."""
    n_dims = coordinates.shape[1]
    
    if n_dims == 1:
        # 2x2 layout: Top row = PES + Persistence, Bottom row = H0 barcodes + H1 barcodes
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Define vibrant colors for each feature
        feature_colors = ['#FF0066', '#0066FF', '#00FF66', '#FF6600', '#6600FF', 
                         '#FFFF00', '#FF00FF', '#00FFFF', '#FF3333', '#33FF33']
        
        # Panel 1: PES with colored feature points
        ax1.plot(coordinates.flatten(), rel_energies, 'k-', alpha=0.4, linewidth=1)
        ax1.scatter(coordinates.flatten(), rel_energies, c='lightgray', s=30, alpha=0.6)
        
        # Track features for color coordination
        feature_data = []
        feature_counter = 0
        
        # Process all generators to assign colors and extract birth/death info
        for dim_key, gen_list in generators.items():
            if not gen_list:
                continue
                
            dim = int(dim_key[1])
            
            for i, gen in enumerate(gen_list):
                birth_simplex = gen.get('birth_simplex', None)
                birth_time = gen['birth']
                death_time = gen['death']
                
                # Determine if spurious
                is_spurious = (birth_time < -0.01 or 
                             (death_time != float('inf') and death_time > rel_energies.max() * 1.1))
                
                # Assign color
                color = feature_colors[feature_counter % len(feature_colors)]
                feature_counter += 1
                
                # Store feature info for persistence diagram
                feature_data.append({
                    'dim': dim,
                    'birth': birth_time,
                    'death': death_time,
                    'color': color,
                    'is_spurious': is_spurious,
                    'birth_simplex': birth_simplex,
                    'label': f'H{dim}.{i+1}'
                })
                
                # Plot on PES
                if birth_simplex is not None and dim == 0:  # H0: highlight birth vertex
                    vertex = birth_simplex[0]
                    if vertex < len(coordinates):
                        marker = 'x' if is_spurious else 'o'
                        size = 120 if not is_spurious else 100
                        alpha = 0.9 if not is_spurious else 0.5
                        
                        if is_spurious:
                            ax1.scatter(coordinates[vertex, 0], rel_energies[vertex], 
                                       c=color, s=size, marker=marker, alpha=alpha)
                        else:
                            ax1.scatter(coordinates[vertex, 0], rel_energies[vertex], 
                                       c=color, s=size, marker=marker, alpha=alpha,
                                       edgecolors='black', linewidth=1)
                        
                        # Label
                        label = f'H{dim}.{i+1}{"*" if is_spurious else ""}'
                        ax1.annotate(label, (coordinates[vertex, 0], rel_energies[vertex]),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
                
                elif birth_simplex is not None and dim == 1 and len(birth_simplex) == 2:  # H1: edge
                    v1, v2 = birth_simplex
                    if v1 < len(coordinates) and v2 < len(coordinates):
                        # Draw thick line
                        linestyle = '--' if is_spurious else '-'
                        alpha = 0.6 if is_spurious else 0.9
                        
                        ax1.plot([coordinates[v1, 0], coordinates[v2, 0]], 
                               [rel_energies[v1], rel_energies[v2]], 
                               color=color, linewidth=4, alpha=alpha, linestyle=linestyle)
                        
                        # Label at midpoint
                        mid_x = (coordinates[v1, 0] + coordinates[v2, 0]) / 2
                        mid_y = (rel_energies[v1] + rel_energies[v2]) / 2
                        label = f'H{dim}.{i+1}{"*" if is_spurious else ""}'
                        ax1.annotate(label, (mid_x, mid_y),
                                   xytext=(5, 5), textcoords='offset points', 
                                   fontsize=10, fontweight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
        
        ax1.set_xlabel('Coordinate', fontsize=12)
        ax1.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
        ax1.set_title(f'{title_prefix} - PES with Topological Features', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Persistence diagram with matching colors
        features = result['features']
        
        # Plot finite features with matching colors
        for feat in feature_data:
            if feat['death'] != float('inf'):
                marker = 'x' if feat['is_spurious'] else 'o'
                size = 80 if not feat['is_spurious'] else 60
                alpha = 0.9 if not feat['is_spurious'] else 0.5
                
                if feat['is_spurious']:
                    ax2.scatter(feat['birth'], feat['death'], 
                               c=feat['color'], s=size, marker=marker, alpha=alpha)
                else:
                    ax2.scatter(feat['birth'], feat['death'], 
                               c=feat['color'], s=size, marker=marker, alpha=alpha,
                               edgecolors='black', linewidth=1)
                
                # Label
                ax2.annotate(feat['label'] + ("*" if feat['is_spurious'] else ""), 
                           (feat['birth'], feat['death']),
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=9, fontweight='bold')
        
        # Plot essential features
        h0_essential = features['h0']['essential']
        if h0_essential:
            print(f"Debug - Essential features: {h0_essential}")
            # Skip essential features for now - causing plotting issues
            # for i, birth in enumerate(h0_essential):
            #     color = feature_colors[(feature_counter + i) % len(feature_colors)]
            #     # Plot essential features at a fixed y-position above the diagram
            #     y_pos = rel_energies.max() * 1.2
            #     ax2.scatter(birth, y_pos, 
            #                c=color, s=120, marker='^', alpha=0.9,
            #                edgecolors='black', linewidth=1, label='Essential' if i == 0 else "")
            #     ax2.annotate(f'H0.ess{i+1}', (birth, y_pos),
            #                xytext=(3, 3), textcoords='offset points', 
            #                fontsize=9, fontweight='bold')
        
        # Add diagonal line
        all_births = [f['birth'] for f in feature_data if f['death'] != float('inf')]
        all_deaths = [f['death'] for f in feature_data if f['death'] != float('inf')]
        
        if all_births and all_deaths:
            max_val = max(max(all_births), max(all_deaths))
            min_val = min(min(all_births), min(all_deaths))
            ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
        
        ax2.set_xlabel('Birth', fontsize=12)
        ax2.set_ylabel('Death', fontsize=12)
        ax2.set_title(f'{title_prefix} - Persistence Diagram', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend for spurious features (no background box)
        ax2.text(0.02, 0.98, '* = spurious feature\n(extended filtration artifact)', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        
        # Panel 3: H0 Barcode diagram
        h0_features = [f for f in feature_data if f['dim'] == 0]
        y_position = 0
        bar_height = 0.6
        
        # Plot H0 finite features
        for feat in h0_features:
            if feat['death'] != float('inf'):
                start = feat['birth']
                end = feat['death']
                
                alpha = 0.6 if feat['is_spurious'] else 0.9
                
                ax3.barh(y_position, end - start, left=start, height=bar_height,
                        color=feat['color'], alpha=alpha, edgecolor='black', linewidth=0.5)
                
                # Add feature label with bigger text
                label = feat['label'] + ("*" if feat['is_spurious'] else "")
                ax3.text(start + (end - start) / 2, y_position, label,
                        ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        # Plot H0 essential features
        h0_essential = features['h0']['essential']
        if h0_essential:
            plot_max = max([f['death'] for f in feature_data if f['death'] != float('inf')] + [rel_energies.max()])
            for i, (birth, death) in enumerate(h0_essential):
                color = feature_colors[(feature_counter + i) % len(feature_colors)]
                
                ax3.barh(y_position, plot_max - birth, left=birth, height=bar_height,
                        color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
                
                # Add arrow to indicate infinity
                ax3.annotate('', xy=(plot_max, y_position + bar_height/2), 
                           xytext=(plot_max * 0.9, y_position + bar_height/2),
                           arrowprops=dict(arrowstyle='->', color=color, lw=3))
                
                # Add label with bigger text
                ax3.text(birth + (plot_max - birth) / 2, y_position, f'H0.ess{i+1}',
                        ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        ax3.set_xlabel('Energy Scale', fontsize=12)
        ax3.set_ylabel('H₀ Features', fontsize=12)
        ax3.set_title(f'{title_prefix} - H₀ Barcodes (Connected Components)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-0.5, max(0.5, y_position - 0.5))
        ax3.set_yticks([])
        
        # Panel 4: H1 Barcode diagram
        h1_features = [f for f in feature_data if f['dim'] == 1]
        y_position = 0
        
        # Plot H1 finite features
        for feat in h1_features:
            if feat['death'] != float('inf'):
                start = feat['birth']
                end = feat['death']
                
                alpha = 0.6 if feat['is_spurious'] else 0.9
                
                ax4.barh(y_position, end - start, left=start, height=bar_height,
                        color=feat['color'], alpha=alpha, edgecolor='black', linewidth=0.5)
                
                # Add feature label with bigger text
                label = feat['label'] + ("*" if feat['is_spurious'] else "")
                ax4.text(start + (end - start) / 2, y_position, label,
                        ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        ax4.set_xlabel('Energy Scale', fontsize=12)
        ax4.set_ylabel('H₁ Features', fontsize=12)
        ax4.set_title(f'{title_prefix} - H₁ Barcodes (Loops/Cycles)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(-0.5, max(0.5, y_position - 0.5))
        ax4.set_yticks([])
        
        # Add note about spurious H1 features in 1D
        if h1_features:
            ax4.text(0.02, 0.98, 'Note: H₁ features in 1D data\nare likely spurious artifacts', 
                    transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        
        plt.suptitle(f'{title_prefix} - Topological Data Analysis\n(Colors match across all plots)', 
                     fontsize=16, fontweight='bold')
        
    else:
        # For higher dimensions, create appropriate layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Basic PES visualization based on dimensionality
        if n_dims == 2:
            scatter = ax1.scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=rel_energies, cmap='viridis', s=20)
            ax1.set_xlabel('Coordinate 1')
            ax1.set_ylabel('Coordinate 2')
            plt.colorbar(scatter, ax=ax1, label='Relative Energy')
        else:  # 3D or higher
            scatter = ax1.scatter(coordinates[:, 0], coordinates[:, 1], 
                               c=rel_energies, cmap='viridis', s=20)
            ax1.set_xlabel('Coordinate 1')
            ax1.set_ylabel('Coordinate 2')
            plt.colorbar(scatter, ax=ax1, label='Relative Energy')
        
        ax1.set_title(f'{title_prefix} - PES ({n_dims}D data)')
        
        # Add other panels as needed...
        ax2.text(0.5, 0.5, f'Generators visualization\nfor {n_dims}D data', 
                ha='center', va='center', transform=ax2.transAxes)
        ax3.text(0.5, 0.5, f'Persistence diagram\nfor {n_dims}D data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax4.text(0.5, 0.5, f'Summary for {n_dims}D data', 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig

def plot_pes_data(coordinates, rel_energies, title_prefix=""):
    """Create appropriate PES visualization based on dimensionality."""
    n_dims = coordinates.shape[1]
    
    if n_dims == 1:
        # 1D: Simple line plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(coordinates.flatten(), rel_energies, 'b-o', markersize=3)
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Relative Energy (kcal/mol)')
        ax.set_title(f'{title_prefix}Potential Energy Surface (1D)')
        ax.grid(True, alpha=0.3)
        
    elif n_dims == 2:
        # 2D: 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot with energy as height
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], rel_energies, 
                           c=rel_energies, cmap='viridis', s=20)
        
        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        ax.set_zlabel('Relative Energy (kcal/mol)')
        ax.set_title(f'{title_prefix}Potential Energy Surface (2D)')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Relative Energy (kcal/mol)')
        
    elif n_dims == 3:
        # 3D: Show slice at minimum energy z-value
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Find slice at minimum energy
        min_idx = np.argmin(rel_energies)
        z_slice = coordinates[min_idx, 2]
        
        # Get points near this slice
        slice_mask = np.abs(coordinates[:, 2] - z_slice) < 0.1 * np.std(coordinates[:, 2])
        slice_coords = coordinates[slice_mask]
        slice_energies = rel_energies[slice_mask]
        
        # 2D projection of slice
        scatter1 = ax1.scatter(slice_coords[:, 0], slice_coords[:, 1], 
                             c=slice_energies, cmap='viridis', s=30)
        ax1.set_xlabel('Coordinate 1')
        ax1.set_ylabel('Coordinate 2')
        ax1.set_title(f'{title_prefix}PES Slice at Z={z_slice:.3f}')
        plt.colorbar(scatter1, ax=ax1, label='Relative Energy')
        
        # 3D scatter plot
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2],
                             c=rel_energies, cmap='viridis', s=10)
        ax2.set_xlabel('Coordinate 1')
        ax2.set_ylabel('Coordinate 2')
        ax2.set_zlabel('Coordinate 3')
        ax2.set_title(f'{title_prefix}3D Point Cloud')
        
    else:
        # Higher dimensions: project to first 2 dimensions
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], 
                           c=rel_energies, cmap='viridis', s=20)
        ax.set_xlabel('Coordinate 1')
        ax.set_ylabel('Coordinate 2')
        ax.set_title(f'{title_prefix}PES Projection to 2D ({n_dims}D data)')
        plt.colorbar(scatter, ax=ax, label='Relative Energy')
        
    plt.tight_layout()
    return fig

def plot_persistence_analysis(result, title_prefix=""):
    """Create persistence diagrams and barcodes with manual plotting and spurious feature warnings."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    features = result['features']
    
    # Persistence diagram for H0
    h0_finite = features['h0']['finite']
    h0_essential = features['h0']['essential']
    
    # Separate valid from spurious H0 features
    h0_valid = []
    h0_spurious = []
    for birth, death in h0_finite:
        if birth < -0.01 or death < -0.01:
            h0_spurious.append((birth, death))
        else:
            h0_valid.append((birth, death))
    
    if h0_valid:
        births, deaths = zip(*h0_valid)
        ax1.scatter(births, deaths, c='red', alpha=0.7, label=f'H₀ valid ({len(h0_valid)})', s=50)
    
    if h0_spurious:
        births, deaths = zip(*h0_spurious)
        ax1.scatter(births, deaths, c='red', alpha=0.3, marker='x', label=f'H₀ spurious ({len(h0_spurious)})', s=50)
    
    # Essential H0 features
    h0_essential_valid = []
    h0_essential_spurious = []
    for birth, _ in h0_essential:
        if birth < -0.01:
            h0_essential_spurious.append(birth)
        else:
            h0_essential_valid.append(birth)
    
    if h0_essential_valid:
        y_pos = max([d for _, d in h0_valid] + h0_essential_valid) * 1.1 if h0_valid else max(h0_essential_valid) * 1.1
        ax1.scatter(h0_essential_valid, [y_pos] * len(h0_essential_valid), c='red', marker='^', s=100, 
                   label=f'H₀ essential valid ({len(h0_essential_valid)})')
    
    if h0_essential_spurious:
        y_pos = max([d for _, d in h0_valid] + h0_essential_spurious) * 1.1 if h0_valid else max(h0_essential_spurious) * 1.1
        ax1.scatter(h0_essential_spurious, [y_pos] * len(h0_essential_spurious), c='red', marker='^', 
                   alpha=0.3, s=100, label=f'H₀ essential spurious ({len(h0_essential_spurious)})')
    
    # Add diagonal line for valid features
    if h0_valid:
        births, deaths = zip(*h0_valid)
        max_val = max(max(births), max(deaths))
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    ax1.set_xlabel('Birth')
    ax1.set_ylabel('Death')
    ax1.set_title(f'{title_prefix}H₀ Persistence Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Persistence diagram for H1
    h1_finite = features['h1']['finite']
    h1_essential = features['h1']['essential']
    
    # Separate valid from spurious H1 features  
    h1_valid = []
    h1_spurious = []
    for birth, death in h1_finite:
        if birth < -0.01 or death < -0.01:
            h1_spurious.append((birth, death))
        else:
            h1_valid.append((birth, death))
    
    if h1_valid:
        births, deaths = zip(*h1_valid)
        ax2.scatter(births, deaths, c='blue', alpha=0.7, label=f'H₁ valid ({len(h1_valid)})', s=50)
        max_val = max(max(births), max(deaths))
        ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    if h1_spurious:
        births, deaths = zip(*h1_spurious)
        ax2.scatter(births, deaths, c='blue', alpha=0.3, marker='x', label=f'H₁ spurious ({len(h1_spurious)})', s=50)
    
    ax2.set_xlabel('Birth')
    ax2.set_ylabel('Death')
    ax2.set_title(f'{title_prefix}H₁ Persistence Diagram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Barcode for H0
    all_h0_valid = h0_valid + [(b, max([d for _, d in h0_valid]) * 1.2) for b in h0_essential_valid] if h0_valid else []
    all_h0_spurious = h0_spurious + [(b, max([d for _, d in h0_spurious]) * 1.2) for b in h0_essential_spurious] if h0_spurious else []
    
    y_pos = 0
    for i, (birth, death) in enumerate(all_h0_valid):
        ax3.barh(y_pos, death - birth, left=birth, height=0.8, color='red', alpha=0.7)
        y_pos += 1
    
    for i, (birth, death) in enumerate(all_h0_spurious):
        ax3.barh(y_pos, death - birth, left=birth, height=0.8, color='red', alpha=0.3, linestyle='--')
        y_pos += 1
    
    ax3.set_xlabel('Energy')
    ax3.set_ylabel('Feature Index')
    ax3.set_title(f'{title_prefix}H₀ Persistence Barcode')
    ax3.grid(True, alpha=0.3)
    
    # Barcode for H1
    y_pos = 0
    for i, (birth, death) in enumerate(h1_valid):
        ax4.barh(y_pos, death - birth, left=birth, height=0.8, color='blue', alpha=0.7)
        y_pos += 1
    
    for i, (birth, death) in enumerate(h1_spurious):
        ax4.barh(y_pos, death - birth, left=birth, height=0.8, color='blue', alpha=0.3, linestyle='--')
        y_pos += 1
    
    ax4.set_xlabel('Energy')
    ax4.set_ylabel('Feature Index')
    ax4.set_title(f'{title_prefix}H₁ Persistence Barcode')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def validate_features(features, coordinates, rel_energies):
    """Validate persistence features and flag potentially spurious ones."""
    print("\n" + "="*60)
    print("FEATURE VALIDATION ANALYSIS")
    print("="*60)
    
    n_dims = coordinates.shape[1]
    energy_range = rel_energies.max() - rel_energies.min()
    persistence_threshold = 0.05 * energy_range  # 5% threshold
    
    validation_results = {}
    
    for dim_key, feature_data in features.items():
        dim = int(dim_key[1])
        finite_features = feature_data['finite']
        essential_features = feature_data['essential']
        
        print(f"\nH{dim} Features:")
        print("-" * 30)
        
        valid_features = []
        spurious_features = []
        
        # Check finite features
        for i, (birth, death) in enumerate(finite_features):
            persistence = death - birth
            issues = []
            
            # Check 1: Extended filtration artifacts
            if birth < -0.01 or death > rel_energies.max() * 1.1:
                issues.append("extended_filtration_artifact")
            
            # Check 2: Geometric inconsistency
            if dim == 1 and n_dims == 1:
                issues.append("impossible_topology_1D")
            elif dim == 2 and n_dims <= 2:
                issues.append("impossible_topology_low_dim")
            
            # Check 3: Very short persistence
            if persistence < persistence_threshold:
                issues.append("short_persistence")
            
            # Check 4: Energy range validation
            if birth < 0 or death < 0:
                issues.append("negative_energy")
            
            feature_info = {
                'birth': birth,
                'death': death,
                'persistence': persistence,
                'issues': issues
            }
            
            if issues:
                spurious_features.append(feature_info)
                status = "⚠️  LIKELY SPURIOUS"
                issue_str = ", ".join(issues)
                print(f"  Feature {i+1}: [{birth:.6f}, {death:.6f}] {status}")
                print(f"    Issues: {issue_str}")
            else:
                valid_features.append(feature_info)
                print(f"  Feature {i+1}: [{birth:.6f}, {death:.6f}] ✅ VALID")
        
        # Check essential features
        for i, (birth, _) in enumerate(essential_features):
            issues = []
            
            if birth < -0.01:
                issues.append("extended_filtration_artifact")
            
            feature_info = {
                'birth': birth,
                'death': float('inf'),
                'persistence': float('inf'),
                'issues': issues
            }
            
            if issues:
                spurious_features.append(feature_info)
                status = "⚠️  LIKELY SPURIOUS"
                issue_str = ", ".join(issues)
                print(f"  Essential {i+1}: [{birth:.6f}, ∞] {status}")
                print(f"    Issues: {issue_str}")
            else:
                valid_features.append(feature_info)
                print(f"  Essential {i+1}: [{birth:.6f}, ∞] ✅ VALID")
        
        validation_results[dim_key] = {
            'valid': valid_features,
            'spurious': spurious_features
        }
    
    # Summary recommendations
    print(f"\n" + "="*60)
    print("VALIDATION SUMMARY & RECOMMENDATIONS")
    print("="*60)
    
    total_valid = sum(len(v['valid']) for v in validation_results.values())
    total_spurious = sum(len(v['spurious']) for v in validation_results.values())
    
    print(f"Valid features: {total_valid}")
    print(f"Likely spurious features: {total_spurious}")
    
    # Specific recommendations
    if n_dims == 1:
        print(f"\nFor 1D PES data, expect:")
        print(f"- 1 essential H0 feature (connected landscape)")
        print(f"- Short-lived H0 features from local minima/saddles")
        print(f"- NO H1 features (1D curves can't have holes)")
        print(f"- NO H2 features (no 3D cavities in 1D)")
        
        h1_count = len(features['h1']['finite']) + len(features['h1']['essential'])
        if h1_count > 0:
            print(f"⚠️  WARNING: Found {h1_count} H1 features - likely Alpha complex artifacts")
            print(f"   Consider using Rips complex instead")
    
    print(f"\nPersistence threshold used: {persistence_threshold:.6f}")
    print(f"Features below this threshold may be noise")
    
    return validation_results

def print_generators(generators, coordinates):
    """Print generator information in a readable format."""
    print("\n" + "="*60)
    print("PERSISTENCE FEATURE GENERATORS")
    print("="*60)
    
    for dim_key, gen_list in generators.items():
        if not gen_list:
            continue
            
        dim = int(dim_key[1])
        dim_name = {0: "Connected Components", 1: "Loops/Cycles", 2: "Voids/Cavities"}
        
        print(f"\n{dim_name.get(dim, f'Dimension {dim}')} (H{dim}):")
        print("-" * 40)
        
        for i, gen in enumerate(gen_list):
            birth = gen['birth']
            death = gen['death']
            cycle = gen.get('cycle', None)
            
            print(f"  Feature {i+1}:")
            print(f"    Birth: {birth:.6f}")
            print(f"    Death: {death:.6f}")
            print(f"    Persistence: {death - birth:.6f}")
            
            if cycle is not None and len(cycle) > 0:
                if dim == 0:
                    # H0: Connected components - show representative vertex
                    print(f"    Representative cycle: {cycle[:5]}{'...' if len(cycle) > 5 else ''}")
                    
                elif dim == 1:
                    # H1: Loops - show cycle of edges
                    print(f"    Generator cycle ({len(cycle)} simplices):")
                    for j, simplex in enumerate(cycle[:3]):  # Show first few simplices
                        if len(simplex) == 2:  # Edge
                            v1, v2 = simplex
                            if v1 < len(coordinates) and v2 < len(coordinates):
                                coord1_str = ", ".join(f"{x:.3f}" for x in coordinates[v1])
                                coord2_str = ", ".join(f"{x:.3f}" for x in coordinates[v2])
                                print(f"      Edge {v1}-{v2}: ({coord1_str}) -- ({coord2_str})")
                            else:
                                print(f"      Edge {v1}-{v2}")
                    if len(cycle) > 3:
                        print(f"      ... and {len(cycle) - 3} more edges")
                        
                elif dim == 2:
                    # H2: Voids - show boundary triangles
                    print(f"    Generator cycle ({len(cycle)} triangles):")
                    for j, simplex in enumerate(cycle[:3]):
                        if len(simplex) == 3:  # Triangle
                            print(f"      Triangle: {simplex}")
                    if len(cycle) > 3:
                        print(f"      ... and {len(cycle) - 3} more triangles")
            else:
                print(f"    Generator cycle: Not available")
            
            print()  # Empty line between features

def main():
    if len(sys.argv) != 2:
        print("Usage: python ph.py <datafile>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not Path(filename).exists():
        print(f"Error: File {filename} not found")
        sys.exit(1)
    
    print(f"Analyzing PES data from: {filename}")
    print("=" * 60)
    
    # Load data
    coordinates, rel_energies = load_pes_data(filename)
    if coordinates is None:
        sys.exit(1)
    
    print(f"Loaded {len(coordinates)} points in {coordinates.shape[1]}D")
    print(f"Energy range: {rel_energies.min():.3f} to {rel_energies.max():.3f} kcal/mol")
    
    # Compute persistence - use appropriate method based on dimensionality
    if coordinates.shape[1] == 1:
        print("\nNote: For 1D data, you may want to use Rips complex instead of Alpha")
        print("Alpha complex result:")
    
    result = compute_alpha_sublevel_main(coordinates, rel_energies)
    if result is None:
        sys.exit(1)
    
    # Validate features for spurious detection
    validation_results = validate_features(result['features'], coordinates, rel_energies)
    
    # Extract generators
    print("\nExtracting generators...")
    generators = extract_generators(result['simplex_tree'], result['features'])
    
    # Create output filename base
    base_name = Path(filename).stem
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Combined analysis figure
    print("Creating combined TDA analysis...")
    combined_fig = create_combined_analysis(coordinates, rel_energies, generators, result, f"{base_name}")
    combined_filename = f"figures/{base_name}_tda_analysis.png"
    combined_fig.savefig(combined_filename, dpi=300, bbox_inches='tight')
    print(f"   ✓ Combined TDA analysis saved to {combined_filename}")
    
    # Individual plots for reference
    # PES plot
    pes_fig = plot_pes_data(coordinates, rel_energies, f"{base_name} - ")
    pes_filename = f"figures/{base_name}_pes.png"
    pes_fig.savefig(pes_filename, dpi=300, bbox_inches='tight')
    
    # PES with generators
    gen_fig = plot_pes_with_generators(coordinates, rel_energies, generators, f"{base_name} - ")
    gen_filename = f"figures/{base_name}_pes_with_generators.png"
    gen_fig.savefig(gen_filename, dpi=300, bbox_inches='tight')
    
    # Persistence analysis
    persist_fig = plot_persistence_analysis(result, f"{base_name} - ")
    persist_filename = f"figures/{base_name}_persistence.png"
    persist_fig.savefig(persist_filename, dpi=300, bbox_inches='tight')
    
    print(f"   ✓ Individual plots also saved")
    
    # Print generators
    print("\nPrinting generators...")
    print_generators(generators, coordinates)
    
    print("\n" + "="*60)
    print(f"Analysis complete for {filename}")
    print(f"Figures saved to:")
    print(f"  - PES: {pes_filename}")
    print(f"  - PES with generators: {gen_filename}")
    print(f"  - Persistence analysis: {persist_filename}")
    print("="*60)

if __name__ == "__main__":
    main() 
