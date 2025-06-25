#!/usr/bin/env python3
"""
Advanced Sublevel Filtration Utilities for Molecular PES Data
Contains comparison methods and advanced analysis tools.
"""

import numpy as np
import matplotlib.pyplot as plt
import gudhi
from scipy.spatial.distance import pdist
from scipy.interpolate import RBFInterpolator
from pathlib import Path

# Set matplotlib backend for headless environments
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
        if coordinates.shape[1] == 0:  # Single column data
            coordinates = data[:, [0]]
        energies = data[:, -1]
        
        # Convert to relative energies
        rel_energies = energies - energies.min()
        
        return coordinates, rel_energies
        
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None

def compute_rips_sublevel_filtration(coordinates, rel_energies):
    """Compute sublevel filtration using Rips complex."""
    print("Computing RIPS sublevel filtration...")
    
    # Estimate appropriate max edge length
    distances = pdist(coordinates)
    max_edge_length = np.percentile(distances, 75)
    
    rips_complex = gudhi.RipsComplex(points=coordinates.tolist(), max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # Assign energy function values
    for i, energy in enumerate(rel_energies):
        simplex_tree.assign_filtration([i], energy)
    
    # Extend to sublevel filtration
    simplex_tree.extend_filtration()
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Extract features
    sublevel_h0 = [(p[1][0], p[1][1]) for p in persistence if p[0] == 0 and p[1][1] != float('inf')]
    sublevel_h1 = [(p[1][0], p[1][1]) for p in persistence if p[0] == 1 and p[1][1] != float('inf')]
    essential_h0 = [(p[1][0], float('inf')) for p in persistence if p[0] == 0 and p[1][1] == float('inf')]
    
    result = {
        'persistence': persistence,
        'sublevel_h0': sublevel_h0,
        'sublevel_h1': sublevel_h1,
        'essential_h0': essential_h0,
        'simplex_tree': simplex_tree
    }
    
    print(f"   ✓ H0 features: {len(sublevel_h0) + len(essential_h0)} (essential: {len(essential_h0)})")
    print(f"   ✓ H1 features: {len(sublevel_h1)}")
    
    return result

def compute_density_based_filtration(coordinates, rel_energies, bandwidth=0.1):
    """Density-based filtration using kernel smoothing."""
    print(f"Computing DENSITY-BASED filtration with bandwidth {bandwidth}...")
    
    n_dims = coordinates.shape[1]
    
    # Create kernel density estimate of energy function
    if n_dims == 1:
        # For 1D data, use simple Gaussian kernel
        coord_range = np.linspace(coordinates.min(), coordinates.max(), 100)
        smooth_energies = np.zeros_like(coord_range)
        
        for i, coord in enumerate(coord_range):
            weights = np.exp(-0.5 * ((coordinates.flatten() - coord) / bandwidth) ** 2)
            weight_sum = weights.sum()
            if weight_sum > 0:
                weights /= weight_sum
                smooth_energies[i] = np.sum(weights * rel_energies)
            else:
                smooth_energies[i] = 0.0
        
        # Create new point cloud from smoothed data
        smooth_coords = coord_range.reshape(-1, 1)
        
    else:
        # For multi-D data, use RBF interpolation
        try:
            rbf = RBFInterpolator(coordinates, rel_energies, smoothing=bandwidth)
            smooth_coords = coordinates.copy()
            smooth_energies = rbf(smooth_coords)
        except:
            # Fallback to original data
            smooth_coords = coordinates.copy()
            smooth_energies = rel_energies.copy()
    
    # Apply sublevel filtration to smoothed data
    smooth_distances = pdist(smooth_coords)
    smooth_max_edge = np.percentile(smooth_distances, 85)
    
    rips_complex = gudhi.RipsComplex(points=smooth_coords.tolist(), max_edge_length=smooth_max_edge)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    
    # Assign smoothed energy values - limit to original data size
    for i, energy in enumerate(smooth_energies[:len(rel_energies)]):
        simplex_tree.assign_filtration([i], energy)
    
    simplex_tree.extend_filtration()
    persistence = simplex_tree.persistence()
    
    # Extract features
    sublevel_h0 = [(p[1][0], p[1][1]) for p in persistence if p[0] == 0 and p[1][1] != float('inf')]
    sublevel_h1 = [(p[1][0], p[1][1]) for p in persistence if p[0] == 1 and p[1][1] != float('inf')]
    essential_h0 = [(p[1][0], float('inf')) for p in persistence if p[0] == 0 and p[1][1] == float('inf')]
    
    result = {
        'persistence': persistence,
        'sublevel_h0': sublevel_h0,
        'sublevel_h1': sublevel_h1,
        'essential_h0': essential_h0,
        'simplex_tree': simplex_tree
    }
    
    print(f"   ✓ H0 features: {len(sublevel_h0) + len(essential_h0)} (essential: {len(essential_h0)})")
    print(f"   ✓ H1 features: {len(sublevel_h1)}")
    
    return result

def compute_alpha_sublevel(coordinates, rel_energies):
    """Compute sublevel filtration using Alpha complex."""
    print(f"Computing Alpha sublevel filtration for {coordinates.shape[1]}D data...")
    
    n_dims = coordinates.shape[1]
    
    if n_dims == 1:
        print("   Warning: Alpha complex in 1D may create spurious H1 features")
        print("   Consider using Rips complex for 1D PES data")
    
    try:
        # Create Alpha complex
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
        
        # Extract generators
        generators = extract_generators(simplex_tree, persistence)
        
        result = {
            'features': features,
            'generators': generators,
            'method': 'alpha',
            'complex_info': f"{simplex_tree.num_vertices()} vertices, {simplex_tree.num_simplices()} simplices"
        }
        
        return result
        
    except Exception as e:
        print(f"Error computing Alpha complex: {e}")
        return None

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

def compare_methods(filename):
    """Compare all sublevel filtration methods."""
    coordinates, rel_energies = load_pes_data(filename)
    if coordinates is None:
        return None
    
    print("Comparing sublevel filtration methods...")
    
    results = {}
    
    # Alpha method
    try:
        if coordinates.shape[1] <= 3:
            alpha_complex = gudhi.AlphaComplex(points=coordinates.tolist())
            simplex_tree = alpha_complex.create_simplex_tree()
            
            for i, energy in enumerate(rel_energies):
                simplex_tree.assign_filtration([i], energy)
            
            simplex_tree.extend_filtration()
            persistence = simplex_tree.persistence()
            
            results['alpha'] = {
                'persistence': persistence,
                'sublevel_h0': [(p[1][0], p[1][1]) for p in persistence if p[0] == 0 and p[1][1] != float('inf')],
                'sublevel_h1': [(p[1][0], p[1][1]) for p in persistence if p[0] == 1 and p[1][1] != float('inf')],
                'essential_h0': [(p[1][0], float('inf')) for p in persistence if p[0] == 0 and p[1][1] == float('inf')]
            }
    except Exception as e:
        print(f"Alpha method failed: {e}")
    
    # Rips method
    try:
        results['rips'] = compute_rips_sublevel_filtration(coordinates, rel_energies)
    except Exception as e:
        print(f"Rips method failed: {e}")
    
    # Density method
    try:
        results['density'] = compute_density_based_filtration(coordinates, rel_energies)
    except Exception as e:
        print(f"Density method failed: {e}")
    
    return results

# Preset functions for specific molecules
def analyze_nh3(filename="data/nh3_pes.dat"):
    """Analyze NH3 pyramidal inversion data."""
    return compare_methods(filename)

def analyze_butane(filename="data/butane_v3.dat"):
    """Analyze butane conformational data."""
    return compare_methods(filename)

class TDAVisualizer:
    """
    Modular visualization system for any TDA method.
    
    Standardizes different persistence computation outputs into a common format
    and provides consistent visualization across all methods.
    """
    
    def __init__(self, coordinates, rel_energies, method_name=""):
        self.coordinates = coordinates
        self.rel_energies = rel_energies
        self.method_name = method_name
        self.n_dims = coordinates.shape[1]
        
        # Vibrant colors for features
        self.feature_colors = ['#FF0066', '#0066FF', '#00FF66', '#FF6600', '#6600FF', 
                              '#FFFF00', '#FF00FF', '#00FFFF', '#FF3333', '#33FF33']
    
    def standardize_result(self, raw_result, method_type):
        """
        Convert any persistence computation result to standard format.
        
        Args:
            raw_result: Output from any persistence method
            method_type: 'alpha', 'rips', 'cubical', 'density', etc.
        
        Returns:
            dict: Standardized format with 'features' and 'generators'
        """
        if method_type == 'alpha':
            # Already in standard format from compute_alpha_sublevel
            return raw_result
        
        elif method_type == 'rips':
            # Convert Rips result to standard format
            persistence = raw_result['persistence']
            features = {}
            
            for dim in range(3):
                finite = [(p[1][0], p[1][1]) for p in persistence if p[0] == dim and p[1][1] != float('inf')]
                essential = [(p[1][0], float('inf')) for p in persistence if p[0] == dim and p[1][1] == float('inf')]
                features[f'h{dim}'] = {'finite': finite, 'essential': essential}
            
            # Extract generators if available
            generators = self._extract_generators_from_persistence(raw_result['simplex_tree'], persistence)
            
            return {
                'features': features,
                'generators': generators,
                'method': 'rips',
                'complex_info': f"{raw_result['simplex_tree'].num_vertices()} vertices, {raw_result['simplex_tree'].num_simplices()} simplices"
            }
        
        elif method_type == 'cubical':
            # Convert cubical result format
            features = {}
            for dim in range(3):
                features[f'h{dim}'] = {'finite': [], 'essential': []}
            
            # Process cubical intervals (format may vary)
            # This would need to be adapted based on actual cubical output format
            
            return {
                'features': features,
                'generators': {},
                'method': 'cubical'
            }
        
        else:
            raise ValueError(f"Unsupported method type: {method_type}")
    
    def _extract_generators_from_persistence(self, simplex_tree, persistence):
        """Extract generators from any GUDHI simplex tree."""
        generators = {'h0': [], 'h1': [], 'h2': []}
        
        try:
            # Get persistence pairs (birth/death simplices)
            pairs = simplex_tree.persistence_pairs()
            
            for i, (birth_simplex, death_simplex) in enumerate(pairs):
                # Determine dimension from simplex size
                dim = len(birth_simplex) - 1
                
                if dim <= 2:  # Only handle H0, H1, H2
                    # Find corresponding persistence interval
                    birth_time = simplex_tree.filtration(birth_simplex)
                    death_time = simplex_tree.filtration(death_simplex) if death_simplex else float('inf')
                    
                    generators[f'h{dim}'].append({
                        'birth': birth_time,
                        'death': death_time,
                        'birth_simplex': birth_simplex,
                        'death_simplex': death_simplex
                    })
        
        except Exception as e:
            print(f"Warning: Could not extract generators - {e}")
        
        return generators
    
    def create_comprehensive_analysis(self, result, title_prefix=""):
        """
        Create the standard 2x2 TDA analysis plot for any method.
        
        Args:
            result: Standardized result from standardize_result()
            title_prefix: Prefix for plot titles
        
        Returns:
            matplotlib.figure.Figure: The complete analysis figure
        """
        if self.n_dims == 1:
            return self._create_1d_analysis(result, title_prefix)
        elif self.n_dims == 2:
            return self._create_2d_analysis(result, title_prefix)
        else:
            return self._create_nd_analysis(result, title_prefix)
    
    def _create_1d_analysis(self, result, title_prefix):
        """Create 2x2 analysis for 1D data."""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        # Process generators for color coordination
        generators = result.get('generators', {})
        feature_data = []
        feature_counter = 0
        
        for dim_key, gen_list in generators.items():
            if not gen_list:
                continue
                
            dim = int(dim_key[1])
            
            for i, gen in enumerate(gen_list):
                birth_time = gen['birth']
                death_time = gen['death']
                
                # Determine if spurious
                is_spurious = (birth_time < -0.01 or 
                             (death_time != float('inf') and death_time > self.rel_energies.max() * 1.1))
                
                # Assign color
                color = self.feature_colors[feature_counter % len(self.feature_colors)]
                feature_counter += 1
                
                feature_data.append({
                    'dim': dim,
                    'birth': birth_time,
                    'death': death_time,
                    'color': color,
                    'is_spurious': is_spurious,
                    'birth_simplex': gen.get('birth_simplex', None),
                    'label': f'H{dim}.{i+1}'
                })
        
        # Panel 1: PES with colored features
        self._plot_pes_with_features(ax1, feature_data, title_prefix)
        
        # Panel 2: Persistence diagram
        self._plot_persistence_diagram(ax2, result, feature_data, title_prefix)
        
        # Panel 3: H0 Barcodes
        self._plot_h0_barcodes(ax3, result, feature_data, title_prefix)
        
        # Panel 4: H1 Barcodes
        self._plot_h1_barcodes(ax4, result, feature_data, title_prefix)
        
        # Overall title
        method_info = result.get('complex_info', result.get('method', ''))
        plt.suptitle(f'{title_prefix} - Topological Data Analysis ({self.method_name})\n{method_info}', 
                     fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_pes_with_features(self, ax, feature_data, title_prefix):
        """Plot PES with colored topological features."""
        # Basic PES
        ax.plot(self.coordinates.flatten(), self.rel_energies, 'k-', alpha=0.4, linewidth=1)
        ax.scatter(self.coordinates.flatten(), self.rel_energies, c='lightgray', s=30, alpha=0.6)
        
        # Plot features
        for feat in feature_data:
            birth_simplex = feat.get('birth_simplex', None)
            color = feat['color']
            is_spurious = feat['is_spurious']
            
            if birth_simplex is not None and feat['dim'] == 0:  # H0 features
                vertex = birth_simplex[0]
                if vertex < len(self.coordinates):
                    marker = 'x' if is_spurious else 'o'
                    size = 120 if not is_spurious else 100
                    alpha = 0.9 if not is_spurious else 0.5
                    
                    if is_spurious:
                        ax.scatter(self.coordinates[vertex, 0], self.rel_energies[vertex], 
                                  c=color, s=size, marker=marker, alpha=alpha)
                    else:
                        ax.scatter(self.coordinates[vertex, 0], self.rel_energies[vertex], 
                                  c=color, s=size, marker=marker, alpha=alpha,
                                  edgecolors='black', linewidth=1)
                    
                    # Label
                    label = f'{feat["label"]}{"*" if is_spurious else ""}'
                    ax.annotate(label, (self.coordinates[vertex, 0], self.rel_energies[vertex]),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
            
            elif birth_simplex is not None and feat['dim'] == 1 and len(birth_simplex) == 2:  # H1 features
                v1, v2 = birth_simplex
                if v1 < len(self.coordinates) and v2 < len(self.coordinates):
                    linestyle = '--' if is_spurious else '-'
                    alpha = 0.6 if is_spurious else 0.9
                    
                    ax.plot([self.coordinates[v1, 0], self.coordinates[v2, 0]], 
                           [self.rel_energies[v1], self.rel_energies[v2]], 
                           color=color, linewidth=4, alpha=alpha, linestyle=linestyle)
                    
                    # Label at midpoint
                    mid_x = (self.coordinates[v1, 0] + self.coordinates[v2, 0]) / 2
                    mid_y = (self.rel_energies[v1] + self.rel_energies[v2]) / 2
                    label = f'{feat["label"]}{"*" if is_spurious else ""}'
                    ax.annotate(label, (mid_x, mid_y),
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))
        
        ax.set_xlabel('Coordinate', fontsize=12)
        ax.set_ylabel('Relative Energy (kcal/mol)', fontsize=12)
        ax.set_title(f'{title_prefix} - PES with Topological Features', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_persistence_diagram(self, ax, result, feature_data, title_prefix):
        """Plot persistence diagram with color coordination."""
        # Plot finite features with matching colors
        for feat in feature_data:
            if feat['death'] != float('inf'):
                marker = 'x' if feat['is_spurious'] else 'o'
                size = 80 if not feat['is_spurious'] else 60
                alpha = 0.9 if not feat['is_spurious'] else 0.5
                
                if feat['is_spurious']:
                    ax.scatter(feat['birth'], feat['death'], 
                              c=feat['color'], s=size, marker=marker, alpha=alpha)
                else:
                    ax.scatter(feat['birth'], feat['death'], 
                              c=feat['color'], s=size, marker=marker, alpha=alpha,
                              edgecolors='black', linewidth=1)
                
                # Label
                ax.annotate(feat['label'] + ("*" if feat['is_spurious'] else ""), 
                           (feat['birth'], feat['death']),
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=9, fontweight='bold')
        
        # Add diagonal line
        all_births = [f['birth'] for f in feature_data if f['death'] != float('inf')]
        all_deaths = [f['death'] for f in feature_data if f['death'] != float('inf')]
        
        if all_births and all_deaths:
            max_val = max(max(all_births), max(all_deaths))
            min_val = min(min(all_births), min(all_deaths))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(f'{title_prefix} - Persistence Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend for spurious features
        ax.text(0.02, 0.98, '* = spurious feature\n(extended filtration artifact)', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    def _plot_h0_barcodes(self, ax, result, feature_data, title_prefix):
        """Plot H0 barcode diagram."""
        h0_features = [f for f in feature_data if f['dim'] == 0]
        y_position = 0
        bar_height = 0.6
        
        # Plot H0 finite features
        for feat in h0_features:
            if feat['death'] != float('inf'):
                start = feat['birth']
                end = feat['death']
                alpha = 0.6 if feat['is_spurious'] else 0.9
                
                ax.barh(y_position, end - start, left=start, height=bar_height,
                       color=feat['color'], alpha=alpha, edgecolor='black', linewidth=0.5)
                
                # Add feature label with bigger text
                label = feat['label'] + ("*" if feat['is_spurious'] else "")
                ax.text(start + (end - start) / 2, y_position, label,
                       ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        # Plot H0 essential features
        features = result['features']
        h0_essential = features['h0']['essential']
        if h0_essential:
            plot_max = max([f['death'] for f in feature_data if f['death'] != float('inf')] + [self.rel_energies.max()])
            for i, (birth, death) in enumerate(h0_essential):
                color = self.feature_colors[(len(h0_features) + i) % len(self.feature_colors)]
                
                ax.barh(y_position, plot_max - birth, left=birth, height=bar_height,
                       color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
                
                # Add arrow to indicate infinity
                ax.annotate('', xy=(plot_max, y_position + bar_height/2), 
                           xytext=(plot_max * 0.9, y_position + bar_height/2),
                           arrowprops=dict(arrowstyle='->', color=color, lw=3))
                
                # Add label with bigger text
                ax.text(birth + (plot_max - birth) / 2, y_position, f'H0.ess{i+1}',
                       ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        ax.set_xlabel('Energy Scale', fontsize=12)
        ax.set_ylabel('H₀ Features', fontsize=12)
        ax.set_title(f'{title_prefix} - H₀ Barcodes (Connected Components)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, max(0.5, y_position - 0.5))
        ax.set_yticks([])
    
    def _plot_h1_barcodes(self, ax, result, feature_data, title_prefix):
        """Plot H1 barcode diagram."""
        h1_features = [f for f in feature_data if f['dim'] == 1]
        y_position = 0
        bar_height = 0.6
        
        # Plot H1 finite features
        for feat in h1_features:
            if feat['death'] != float('inf'):
                start = feat['birth']
                end = feat['death']
                alpha = 0.6 if feat['is_spurious'] else 0.9
                
                ax.barh(y_position, end - start, left=start, height=bar_height,
                       color=feat['color'], alpha=alpha, edgecolor='black', linewidth=0.5)
                
                # Add feature label with bigger text
                label = feat['label'] + ("*" if feat['is_spurious'] else "")
                ax.text(start + (end - start) / 2, y_position, label,
                       ha='center', va='center', fontsize=12, fontweight='bold')
                
                y_position += 1
        
        ax.set_xlabel('Energy Scale', fontsize=12)
        ax.set_ylabel('H₁ Features', fontsize=12)
        ax.set_title(f'{title_prefix} - H₁ Barcodes (Loops/Cycles)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, max(0.5, y_position - 0.5))
        ax.set_yticks([])
        
        # Add note about spurious H1 features in 1D
        if h1_features and self.n_dims == 1:
            ax.text(0.02, 0.98, 'Note: H₁ features in 1D data\nare likely spurious artifacts', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    def _create_2d_analysis(self, result, title_prefix):
        """Create analysis for 2D data.""" 
        # Implementation for 2D visualization
        pass
    
    def _create_nd_analysis(self, result, title_prefix):
        """Create analysis for higher dimensional data."""
        # Implementation for nD visualization  
        pass


def create_universal_tda_analysis(coordinates, rel_energies, method_name, raw_result, method_type, output_name):
    """
    Universal function to create TDA analysis for any persistence method.
    
    Args:
        coordinates: Point coordinates
        rel_energies: Energy values
        method_name: Human-readable method name
        raw_result: Raw output from persistence computation
        method_type: Type identifier ('alpha', 'rips', 'cubical', etc.)
        output_name: Base name for output files
    
    Returns:
        dict: Analysis results and figure
    """
    # Create visualizer
    viz = TDAVisualizer(coordinates, rel_energies, method_name)
    
    # Standardize the result format
    standardized_result = viz.standardize_result(raw_result, method_type)
    
    # Create comprehensive analysis
    base_name = output_name.replace('.dat', '').replace('.csv', '')
    fig = viz.create_comprehensive_analysis(standardized_result, base_name)
    
    # Save figure
    import os
    os.makedirs('figures', exist_ok=True)
    filename = f"figures/{base_name}_{method_type}_analysis.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"   ✓ Universal TDA analysis saved to {filename}")
    
    return {
        'figure': fig,
        'standardized_result': standardized_result,
        'filename': filename
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results = compare_methods(sys.argv[1])
        if results:
            for method, result in results.items():
                print(f"\n{method.upper()} method:")
                print(f"  H0: {len(result.get('sublevel_h0', [])) + len(result.get('essential_h0', []))}")
                print(f"  H1: {len(result.get('sublevel_h1', []))}")
    else:
        print("Usage: python ph_utils.py <datafile>") 
