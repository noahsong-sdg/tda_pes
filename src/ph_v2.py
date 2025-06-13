#!/usr/bin/env python3
"""
PES Topological Data Analysis v2 with Fixed 2D Cubical Complex + Generator Tracking

Key outputs:
- PES with generators mapped to conformers
- Persistence barcode with chemical labels
- Persistence diagram
- Generator analysis
"""

import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import gudhi as gd

def load_pes_data(filename: str, coordinate_names: list[str] = None, coordinate_units: list[str] = None):
    """
    Load n-dimensional PES data from file.
    
    Args:
        filename: Path to data file
        coordinate_names: List of names for coordinates (optional, will auto-generate if None)
        coordinate_units: List of units for coordinates (optional, will auto-detect if None)
    
    Returns:
        coordinates: numpy array of shape (n_points, n_dimensions) 
        energies: numpy array of relative energies
        metadata: dict with information about the loaded data
    
    Expected file format:
        # Comments (optional)
        coord1  coord2  ...  coordN  energy
        val1    val2    ...  valN    E1
        ...
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None, None, None
    
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        print(f"Error loading data from {filename}: {e}")
        return None, None, None
    
    # Handle different data shapes
    if data.ndim == 1:
        # Single data point
        data = data.reshape(1, -1)
    
    n_points, n_columns = data.shape
    
    assert n_columns >= 2, "Data must have at least 2 columns (coordinate + energy)"
    
    # Split coordinates and energies
    coordinates = data[:, :-1]  # All columns except last
    energies = data[:, -1]      # Last column is energy
    n_dimensions = coordinates.shape[1]
    
    # Make energies relative to minimum
    rel_energies = energies - np.min(energies)
    
    # Auto-generate coordinate names if not provided
    if coordinate_names is None:
        coordinate_names = []
        for i in range(n_dimensions):
            # Try to guess coordinate type based on value ranges
            coord_values = coordinates[:, i]
            coord_range = np.max(coord_values) - np.min(coord_values)
            coord_max = np.max(np.abs(coord_values))
            
            if coord_max > 180 and coord_range > 180:
                # Likely angle in degrees
                coordinate_names.append(f"angle_{i+1}")
            elif coord_max <= 2*np.pi and coord_range <= 2*np.pi:
                # Likely angle in radians or small distance
                if coord_max > 7:  # > 2π
                    coordinate_names.append(f"coord_{i+1}")
                else:
                    coordinate_names.append(f"angle_{i+1}_rad")
            elif 0.5 <= coord_max <= 5.0:
                # Likely bond length in Angstroms
                coordinate_names.append(f"bond_length_{i+1}")
            else:
                # Generic coordinate
                coordinate_names.append(f"coord_{i+1}")
    
    # Auto-detect coordinate units if not provided
    if coordinate_units is None:
        coordinate_units = []
        for i in range(n_dimensions):
            coord_values = coordinates[:, i]
            coord_max = np.max(np.abs(coord_values))
            
            if coord_max > 180:
                coordinate_units.append("degrees")
            elif coord_max <= 2*np.pi and coord_max > 3:
                coordinate_units.append("radians")
            elif 0.5 <= coord_max <= 5.0:
                coordinate_units.append("Angstrom")
            else:
                coordinate_units.append("unknown")
    
    # Create metadata
    metadata = {
        'n_points': n_points,
        'n_dimensions': n_dimensions,
        'coordinate_names': coordinate_names,
        'coordinate_units': coordinate_units,
        'coordinate_ranges': [],
        'energy_range_hartree': (np.min(rel_energies), np.max(rel_energies)),
        'energy_range_kcal_mol': (np.min(rel_energies) * 627.509, np.max(rel_energies) * 627.509),
        'filename': filename
    }
    
    # Add coordinate ranges
    for i in range(n_dimensions):
        coord_min = np.min(coordinates[:, i])
        coord_max = np.max(coordinates[:, i])
        metadata['coordinate_ranges'].append((coord_min, coord_max))
    
    # Print summary
    print(f"Loaded {n_dimensions}D PES data: {n_points} points")
    print(f"Coordinates:")
    for i, (name, unit, (min_val, max_val)) in enumerate(zip(coordinate_names, coordinate_units, metadata['coordinate_ranges'])):
        print(f"  {name} ({unit}): {min_val:.3f} to {max_val:.3f}")
    print(f"Energy range: {np.min(rel_energies):.6f} to {np.max(rel_energies):.6f} Hartree")
    print(f"Energy range: {np.min(rel_energies)*627.509:.3f} to {np.max(rel_energies)*627.509:.3f} kcal/mol")
    
    return coordinates, rel_energies, metadata


def perform_tda_analysis(coordinates, energies, metadata):
    """
    Perform TDA analysis for n-dimensional PES data.
    Chooses appropriate method based on dimensionality and coordinate types.
    """
    n_dimensions = metadata['n_dimensions']
    n_points = metadata['n_points']
    coordinate_names = metadata['coordinate_names']
    
    print(f"Computing persistence for {n_dimensions}D data with {n_points} points...")
    print(f"Coordinates: {coordinate_names}")
    
    if n_dimensions == 1:
        # 1D case: Use existing methods with periodicity detection
        coord_values = coordinates[:, 0]
        cubical_persistence = compute_sublevel_cubical_persistence_1d(coord_values, energies, metadata)
        simplex_persistence, simplex_tree = compute_simplex_persistence_1d(coord_values, energies)
        
        # Choose best method
        h1_cubical = sum(1 for dim, _ in cubical_persistence if dim == 1)
        h1_simplex = sum(1 for dim, _ in simplex_persistence if dim == 1)
        
        print(f"H1 detection: Cubical={h1_cubical}, Simplex={h1_simplex}")
        
        if h1_cubical > 0 and h1_simplex > 0:
            print("Using Cubical persistence with Simplex generators")
            best_persistence = cubical_persistence
            generator_tree = simplex_tree
        elif h1_simplex > 0:
            print("Using Simplex persistence and generators")
            best_persistence = simplex_persistence
            generator_tree = simplex_tree
        else:
            print("Using Cubical persistence (no H1 found)")
            best_persistence = cubical_persistence
            generator_tree = None
            
    elif n_dimensions == 2:
        # 2D case: Use 2D cubical complex
        print("Using 2D cubical complex approach")
        best_persistence = compute_nd_sublevel_persistence(coordinates, energies, metadata)
        generator_tree = None  # Generator extraction for 2D+ is more complex
        
    else:
        # Higher dimensions: Use n-dimensional cubical complex
        print(f"Using {n_dimensions}D cubical complex approach")
        best_persistence = compute_nd_sublevel_persistence(coordinates, energies, metadata)
        generator_tree = None  # Generator extraction for high-D is more complex
    
    return best_persistence, coordinates, generator_tree


def compute_sublevel_cubical_persistence_1d(angles, energies, metadata):
    """
    Compute persistence using proper sublevel filtration on cubical complex.
    
    This is the correct approach for studying topology of sublevel sets of 
    potential energy functions. The key insight is to use the energy values
    directly as the filtration function on the natural grid structure.
    
    For molecular PES data:
    - 1D conformational scans → 1D cubical complex
    - 2D conformational scans → 2D cubical complex  
    - Higher dimensions → Higher-dimensional cubical complex
    
    References:
    - GUDHI tutorial: https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-cubical-complexes.ipynb
    - Sublevel filtration theory: Studies topology of {x : f(x) ≤ t} as t increases
    """
    print(f"Computing sublevel filtration on 1D cubical complex")
    print(f"Data points: {len(angles)}, Energy range: {np.min(energies):.6f} to {np.max(energies):.6f} Hartree")
    
    # For 1D periodic data (dihedral angles), we need to handle periodicity correctly
    # Sort data by angle for proper grid structure
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_energies = energies[sorted_indices]
    
    # Check if data spans full 360° range (periodic)
    angle_range = np.max(sorted_angles) - np.min(sorted_angles)
    is_periodic = angle_range > 300  # Consider periodic if covers most of 360°
    
    if is_periodic:
        print("Detected periodic data (likely dihedral scan)")
        # Use PeriodicCubicalComplex for proper periodic boundary conditions
        persistence = compute_periodic_1d_persistence(sorted_energies)
    else:
        print("Detected non-periodic data")
        # Use regular CubicalComplex
        persistence = compute_regular_1d_persistence(sorted_energies)
    
    # Apply chemical significance filtering
    chemical_threshold_hartree = 0.2 / 627.509  # 0.2 kcal/mol → Hartree
    print(f"Filtering features below {0.2} kcal/mol ({chemical_threshold_hartree:.6f} Hartree)")
    
    filtered_persistence = []
    filtered_count = 0
    
    for dim, (birth, death) in persistence:
        persistence_val = death - birth if death != float('inf') else float('inf')
        
        # Keep all H0 features (connected components always important)
        if dim == 0:
            filtered_persistence.append((dim, (birth, death)))
        # Keep chemically significant H1 features (loops/cycles)
        elif dim == 1 and persistence_val >= chemical_threshold_hartree:
            filtered_persistence.append((dim, (birth, death)))
        elif dim == 1:
            filtered_count += 1
        # Keep higher-dimensional features if present
        elif dim >= 2:
            filtered_persistence.append((dim, (birth, death)))
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} chemically insignificant H1 features")
    
    h0_count = sum(1 for dim, _ in filtered_persistence if dim == 0)
    h1_count = sum(1 for dim, _ in filtered_persistence if dim == 1)
    h2_count = sum(1 for dim, _ in filtered_persistence if dim == 2)
    
    print(f"Final features: H0={h0_count}, H1={h1_count}, H2={h2_count}")
    
    return filtered_persistence


def compute_periodic_1d_persistence(energies):
    """Compute persistence for periodic 1D data using PeriodicCubicalComplex."""
    try:
        from gudhi import PeriodicCubicalComplex
        
        # For 1D periodic data, specify periodicity in first dimension
        pcc = PeriodicCubicalComplex(
            dimensions=[len(energies)],
            top_dimensional_cells=energies,
            periodic_dimensions=[True]  # First dimension is periodic
        )
        
        persistence = pcc.persistence()
        print(f"Periodic cubical complex: {pcc.num_simplices()} simplices, dimension {pcc.dimension()}")
        
        return persistence
        
    except ImportError:
        print("Warning: PeriodicCubicalComplex not available, falling back to regular complex")
        return compute_regular_1d_persistence(energies)
    except Exception as e:
        print(f"Warning: Periodic computation failed ({e}), falling back to regular complex")
        return compute_regular_1d_persistence(energies)


def compute_regular_1d_persistence(energies):
    """Compute persistence for non-periodic 1D data using regular CubicalComplex."""
    from gudhi import CubicalComplex
    
    # Create 1D cubical complex directly from energy values
    cc = CubicalComplex(
        dimensions=[len(energies)],
        top_dimensional_cells=energies
    )
    
    persistence = cc.persistence()
    print(f"Regular cubical complex: {cc.num_simplices()} simplices, dimension {cc.dimension()}")
    
    return persistence


def compute_2d_sublevel_persistence(energy_grid):
    """
    Compute persistence for 2D energy grids (e.g., two-dimensional conformational scans).
    
    Args:
        energy_grid: 2D numpy array where energy_grid[i,j] is the energy at grid point (i,j)
    
    Returns:
        persistence: Persistence pairs from sublevel filtration
    """
    from gudhi import CubicalComplex
    
    height, width = energy_grid.shape
    print(f"Computing 2D sublevel filtration on {height}x{width} grid")
    
    # Create 2D cubical complex
    cc = CubicalComplex(
        dimensions=[height, width],
        top_dimensional_cells=energy_grid.flatten()
    )
    
    persistence = cc.persistence()
    print(f"2D cubical complex: {cc.num_simplices()} simplices, dimension {cc.dimension()}")
    
    return persistence


def compute_higher_dim_sublevel_persistence(energy_data, dimensions):
    """
    Compute persistence for higher-dimensional conformational spaces.
    
    Args:
        energy_data: Flattened array of energy values
        dimensions: List specifying grid dimensions [dim1, dim2, dim3, ...]
    
    Returns:
        persistence: Persistence pairs from sublevel filtration
    """
    from gudhi import CubicalComplex
    
    print(f"Computing {len(dimensions)}D sublevel filtration on grid {dimensions}")
    
    # Create higher-dimensional cubical complex
    cc = CubicalComplex(
        dimensions=dimensions,
        top_dimensional_cells=energy_data
    )
    
    persistence = cc.persistence()
    print(f"{len(dimensions)}D cubical complex: {cc.num_simplices()} simplices, dimension {cc.dimension()}")
    
    return persistence


def compute_nd_sublevel_persistence(coordinates, energies, metadata):
    """
    Compute n-dimensional sublevel persistence for arbitrary coordinate types.
    
    Args:
        coordinates: numpy array of shape (n_points, n_dimensions)
        energies: numpy array of energy values
        metadata: dict with coordinate information
    
    Returns:
        persistence: Persistence pairs from sublevel filtration
    """
    from gudhi import CubicalComplex
    
    n_dimensions = metadata['n_dimensions']
    n_points = metadata['n_points']
    coordinate_names = metadata['coordinate_names']
    coordinate_units = metadata['coordinate_units']
    
    print(f"Computing {n_dimensions}D sublevel persistence")
    print(f"Data structure: {n_points} points in {n_dimensions}D space")
    
    if n_dimensions == 1:
        # For 1D, use the existing optimized approach
        return compute_sublevel_cubical_persistence_1d(coordinates[:, 0], energies, metadata)
    
    elif n_dimensions == 2:
        # For 2D, try to detect if this is a regular grid
        coord1_vals = coordinates[:, 0]
        coord2_vals = coordinates[:, 1]
        
        # Check if coordinates form a regular grid
        unique_coord1 = np.unique(coord1_vals)
        unique_coord2 = np.unique(coord2_vals)
        
        if len(unique_coord1) * len(unique_coord2) == n_points:
            # Regular grid detected - use 2D cubical complex
            print(f"Regular 2D grid detected: {len(unique_coord1)} x {len(unique_coord2)}")
            
            # Reshape energies into grid
            energy_grid = np.zeros((len(unique_coord1), len(unique_coord2)))
            for i, (c1, c2, e) in enumerate(zip(coord1_vals, coord2_vals, energies)):
                idx1 = np.where(unique_coord1 == c1)[0][0]
                idx2 = np.where(unique_coord2 == c2)[0][0]
                energy_grid[idx1, idx2] = e
            
            return compute_2d_sublevel_persistence(energy_grid)
        else:
            # Irregular grid - use point cloud approach
            print("Irregular 2D data - using point cloud approach")
            return compute_point_cloud_persistence(coordinates, energies, metadata)
    
    else:
        # Higher dimensions - check for regular grid structure
        print(f"Processing {n_dimensions}D data")
        
        # Try to detect regular grid structure
        grid_dimensions = []
        is_regular_grid = True
        
        for dim in range(n_dimensions):
            unique_vals = np.unique(coordinates[:, dim])
            grid_dimensions.append(len(unique_vals))
        
        expected_points = np.prod(grid_dimensions)
        if expected_points == n_points:
            print(f"Regular {n_dimensions}D grid detected: {grid_dimensions}")
            
            # Create flattened energy array in proper order
            # This is complex for arbitrary dimensions, so use simple approach
            return compute_higher_dim_sublevel_persistence(energies, grid_dimensions)
        else:
            # Irregular data - use point cloud approach
            print(f"Irregular {n_dimensions}D data - using point cloud approach")
            return compute_point_cloud_persistence(coordinates, energies, metadata)


def compute_point_cloud_persistence(coordinates, energies, metadata):
    """
    Compute persistence for irregular point cloud data using Rips complex.
    
    Args:
        coordinates: numpy array of shape (n_points, n_dimensions)
        energies: numpy array of energy values
        metadata: dict with coordinate information
    
    Returns:
        persistence: Persistence pairs from sublevel filtration
    """
    try:
        from gudhi import RipsComplex
        from scipy.spatial.distance import pdist, squareform
        
        print("Using Rips complex for irregular point cloud")
        
        # Compute distance matrix in coordinate space
        distances = squareform(pdist(coordinates))
        
        # Create Rips complex with energy-based filtration
        rips = RipsComplex(distance_matrix=distances, max_edge_length=np.inf)
        simplex_tree = rips.create_simplex_tree(max_dimension=min(3, metadata['n_dimensions']))
        
        # Apply energy-based filtration
        for i, energy in enumerate(energies):
            simplex_tree.assign_filtration([i], energy)
        
        # Extend filtration to higher-dimensional simplices
        simplex_tree.extend_filtration()
        
        persistence = simplex_tree.persistence()
        
        print(f"Rips complex: {simplex_tree.num_simplices()} simplices")
        return persistence
        
    except ImportError:
        print("Warning: Advanced point cloud analysis requires scipy")
        # Fallback to simple approach
        print("Using fallback: treating as 1D energy sequence")
        return compute_sublevel_cubical_persistence_1d(np.arange(len(energies)), energies, metadata)


def compute_simplex_persistence_1d(angles, energies):
    """Compute persistence using simplex tree for generator extraction."""
    n = len(energies)
    st = gd.SimplexTree()
    
    # Add vertices with energy as filtration
    for i, energy in enumerate(energies):
        st.insert([i], filtration=energy)
    
    # Add edges
    for i in range(n-1):
        edge_filt = max(energies[i], energies[i+1])
        st.insert([i, i+1], filtration=edge_filt)
    
    # Periodic edge
    periodic_filt = max(energies[0], energies[-1])
    st.insert([0, n-1], filtration=periodic_filt)
    
    # Add triangles (conservative approach)
    for i in range(n):
        j = (i + 1) % n
        k = (i + 2) % n
        triangle_filt = max(energies[i], energies[j], energies[k])
        st.insert([i, j, k], filtration=triangle_filt)
    
    # Compute persistence
    persistence = st.persistence(homology_coeff_field=2, min_persistence=0)
    
    return persistence, st


def analyze_persistence_with_generators(persistence, energies, simplex_tree=None, angles=None):
    """Extract generator information for chemical interpretation."""
    generators_info = []
    
    print("\n--- PERSISTENCE GENERATORS ---")
    h0_count = 0
    h1_count = 0
    
    for i, (dim, (birth, death)) in enumerate(persistence):
        if dim not in [0, 1]:
            continue
        
        persistence_val = death - birth if death != float('inf') else float('inf')
        
        generator_info = {
            'dimension': dim,
            'birth': birth,
            'death': death,
            'persistence': persistence_val,
            'index': i,
            'birth_simplex': None,
            'death_simplex': None,
            'chemical_angles': []
        }
        
        # Extract generator information from simplex tree
        if simplex_tree is not None:
            try:
                persistence_pairs = simplex_tree.persistence_pairs()
                if i < len(persistence_pairs):
                    birth_simplex, death_simplex = persistence_pairs[i]
                    generator_info['birth_simplex'] = birth_simplex
                    generator_info['death_simplex'] = death_simplex
                    
                    # Map vertices to chemical angles
                    if birth_simplex and angles is not None:
                        chemical_angles = []
                        for vertex_idx in birth_simplex:
                            if vertex_idx < len(angles):
                                chemical_angles.append(angles[vertex_idx])
                        generator_info['chemical_angles'] = chemical_angles
            except:
                pass
        
        generators_info.append(generator_info)
        
        if dim == 0:
            h0_count += 1
            print(f"[H0-{h0_count}] Component: birth={birth:.6f}, death={death:.6f}, persistence={persistence_val:.6f}")
            if generator_info['chemical_angles']:
                print(f"    At angles: {generator_info['chemical_angles']}")
        elif dim == 1:
            h1_count += 1
            print(f"[H1-{h1_count}] Cycle: birth={birth:.6f}, death={death:.6f}, persistence={persistence_val:.6f}")
            print(f"    Birth energy: {birth*627.509:.2f} kcal/mol")
            if generator_info['chemical_angles']:
                print(f"    At conformer angles: {[f'{a:.1f}°' for a in generator_info['chemical_angles']]}")
    
    return generators_info


def create_essential_plots(angles, energies, persistence, simplex_tree=None, output_name="pes_analysis"):
    """Create the 4 essential plots with proper generator tracking."""
    os.makedirs("figures", exist_ok=True)
    
    # Extract generators
    generators_info = analyze_persistence_with_generators(persistence, energies, simplex_tree, angles)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    h0_color = 'red'
    h1_color = 'blue'
    
    # Plot 1: PES with generators highlighted
    ax1 = axes[0, 0]
    ax1.plot(angles, energies, 'k-', linewidth=2, alpha=0.7, label='PES')
    ax1.scatter(angles, energies, c='gray', s=30, alpha=0.6, zorder=3)
    
    # Highlight generators
    h1_generator_count = 0
    for gen_info in generators_info:
        if gen_info['dimension'] == 0 and gen_info['birth_simplex'] is not None:
            # H0 generator
            if len(gen_info['birth_simplex']) == 1:
                vertex_idx = gen_info['birth_simplex'][0]
                if vertex_idx < len(angles):
                    ax1.scatter(angles[vertex_idx], energies[vertex_idx], 
                              c=h0_color, s=100, marker='o', zorder=5,
                              edgecolors='black', linewidth=1)
        
        elif gen_info['dimension'] == 1 and gen_info['birth_simplex'] is not None:
            # H1 generator
            h1_generator_count += 1
            vertices = gen_info['birth_simplex']
            max_energy_vertex = None
            max_energy = -float('inf')
            
            for vertex_idx in vertices:
                if vertex_idx < len(angles):
                    ax1.scatter(angles[vertex_idx], energies[vertex_idx], 
                              c=h1_color, s=100, marker='s', zorder=5,
                              edgecolors='black', linewidth=1)
                    
                    if energies[vertex_idx] > max_energy:
                        max_energy = energies[vertex_idx]
                        max_energy_vertex = vertex_idx
            
            # Label the highest energy vertex
            if max_energy_vertex is not None:
                ax1.annotate(f'H1-{h1_generator_count}', 
                           xy=(angles[max_energy_vertex], energies[max_energy_vertex]),
                           xytext=(8, 12), textcoords='offset points',
                           fontsize=10, fontweight='bold', color=h1_color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=h1_color, alpha=0.9),
                           arrowprops=dict(arrowstyle='->', color=h1_color, lw=1.5))
    
    ax1.set_xlabel('Dihedral Angle (degrees)')
    ax1.set_ylabel('Relative Energy (Hartree)')
    ax1.set_title('PES with Generators (Chemical Conformers)')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], color='k', linewidth=2, alpha=0.7, label='PES')]
    if any(g['dimension'] == 0 for g in generators_info):
        legend_elements.append(plt.scatter([], [], c=h0_color, s=100, marker='o', 
                                         edgecolors='black', linewidth=1, label='H0 Generators'))
    if any(g['dimension'] == 1 for g in generators_info):
        legend_elements.append(plt.scatter([], [], c=h1_color, s=100, marker='s', 
                                         edgecolors='black', linewidth=1, label='H1 Generators'))
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Persistence diagram
    ax2 = axes[0, 1]
    
    h0_features = [(birth, death) for dim, (birth, death) in persistence if dim == 0]
    h1_features = [(birth, death) for dim, (birth, death) in persistence if dim == 1]
    
    if h0_features:
        births, deaths = zip(*h0_features)
        ax2.scatter(births, deaths, c=h0_color, s=80, alpha=0.8, label='H0')
    
    if h1_features:
        births, deaths = zip(*h1_features)
        ax2.scatter(births, deaths, c=h1_color, s=80, alpha=0.8, label='H1')
    
    max_val = max(energies) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Birth = Death')
    
    ax2.set_xlabel('Birth (Hartree)')
    ax2.set_ylabel('Death (Hartree)')
    ax2.set_title('Persistence Diagram')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Persistence barcode with chemical labels
    ax3 = axes[1, 0]
    
    y_pos = 0
    bar_height = 0.8
    
    # Plot H0 bars
    for i, (birth, death) in enumerate(h0_features):
        if death == float('inf'):
            death = max(energies) * 1.1
        ax3.barh(y_pos, death - birth, left=birth, height=bar_height, 
                color=h0_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.text(birth + (death - birth) / 2, y_pos, f'H0-{i+1}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
        y_pos += 1
    
    # Plot H1 bars with chemical information
    for i, (birth, death) in enumerate(h1_features):
        if death == float('inf'):
            death = max(energies) * 1.1
        ax3.barh(y_pos, death - birth, left=birth, height=bar_height, 
                color=h1_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add chemical information if available
        label = f'H1-{i+1}'
        if i < len([g for g in generators_info if g['dimension'] == 1]):
            h1_gen = [g for g in generators_info if g['dimension'] == 1][i]
            if h1_gen['chemical_angles']:
                # Show representative angle
                avg_angle = np.mean(h1_gen['chemical_angles'])
                label += f' ({avg_angle:.0f}°)'
        
        ax3.text(birth + (death - birth) / 2, y_pos, label, 
                ha='center', va='center', fontsize=8, fontweight='bold')
        y_pos += 1
    
    ax3.set_xlabel('Energy (Hartree)')
    ax3.set_ylabel('Features')
    ax3.set_title('Persistence Barcode (with Chemical Info)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Chemical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    h0_count = len(h0_features)
    h1_count = len(h1_features)
    h2_count = sum(1 for dim, _ in persistence if dim == 2)
    
    # Create summary with chemical interpretation
    summary_text = f"""TOPOLOGICAL ANALYSIS RESULTS

Feature Counts:
• H0 (Components): {h0_count}
• H1 (Cycles): {h1_count}  
• H2 (Voids): {h2_count}

Energy Scale:
• Range: 0 to {np.max(energies)*627.5:.1f} kcal/mol
• Scale: {np.max(energies):.6f} Hartree

Chemical Interpretation:
• H0 = Energy basins/conformers
• H1 = Conformational transition cycles
• Higher persistence = more significant

H1 Cycle Details:"""
    
    # Add H1 generator details
    h1_generators = [g for g in generators_info if g['dimension'] == 1]
    for i, gen in enumerate(h1_generators[:3]):  # Show first 3
        birth_kcal = gen['birth'] * 627.509
        summary_text += f"\n• H1-{i+1}: {birth_kcal:.1f} kcal/mol barrier"
        if gen['chemical_angles']:
            representative_angles = gen['chemical_angles'][:3]  # Show first 3 angles
            summary_text += f"\n  Involves conformers at: {[f'{a:.0f}°' for a in representative_angles]}"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"figures/{output_name}.png", dpi=300, bbox_inches='tight')
    print(f"Saved analysis to figures/{output_name}.png")


def main():
    """
    Main function demonstrating proper sublevel filtration for molecular PES.
    
    This implementation uses cubical complexes wrt sublevel sets {x : f(x) ≤ t}
    
    Examples of proper usage:
    
    1D Conformational scan (dihedral angle)
        python ph_v2.py data/butane_pes.dat
        
    Key advantages of this approach:
    - Uses natural grid structure of conformational space
    - Proper periodic boundary conditions for dihedral angles
    - Direct sublevel filtration without artificial embeddings
    - Scales naturally to higher dimensions
    
    References:
    - GUDHI cubical tutorial: https://github.com/GUDHI/TDA-tutorial/blob/master/Tuto-GUDHI-cubical-complexes.ipynb
    """
    parser = argparse.ArgumentParser(
        description="PES Topological Analysis v2 with Proper Sublevel Filtration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("filename", help="Path to PES data file (.dat)")
    parser.add_argument("--output", default="pes_analysis_v2", 
                       help="Output file prefix")
    
    args = parser.parse_args()
    
    print("PES Topological Data Analysis v2")
    print("Proper Sublevel Filtration with Cubical Complexes")
    print("="*55)    
    
    # Load data
    coordinates, rel_energies, metadata = load_pes_data(args.filename)
    if coordinates is None:
        print(f"Error: Could not load data from {args.filename}")
        return
    
    print(f"Loaded {len(coordinates)} data points")
    print(f"Coordinates:")
    for i, (name, unit, (min_val, max_val)) in enumerate(zip(metadata['coordinate_names'], metadata['coordinate_units'], metadata['coordinate_ranges'])):
        print(f"  {name} ({unit}): {min_val:.3f} to {max_val:.3f}")
    print(f"Energy range: {np.min(rel_energies):.6f} to {np.max(rel_energies):.6f} Hartree")
    print(f"Energy range: {np.min(rel_energies)*627.509:.3f} to {np.max(rel_energies)*627.509:.3f} kcal/mol")
    
    # Perform proper sublevel filtration analysis
    print("\n--- SUBLEVEL FILTRATION ANALYSIS ---")
    persistence, coordinates_out, simplex_tree = perform_tda_analysis(coordinates, rel_energies, metadata)
    
    # Count features
    h0_count = sum(1 for dim, _ in persistence if dim == 0)
    h1_count = sum(1 for dim, _ in persistence if dim == 1)
    h2_count = sum(1 for dim, _ in persistence if dim >= 2)
    
    print(f"\n--- TOPOLOGICAL FEATURES (SUBLEVEL SETS) ---")
    print(f"H0 (Connected Components): {h0_count}")
    print(f"H1 (Loops/Cycles): {h1_count}")
    print(f"H2+ (Higher-dim features): {h2_count}")
    
    # Create visualizations with proper labeling
    if metadata['n_dimensions'] == 1:
        create_essential_plots(coordinates[:, 0], rel_energies, persistence, simplex_tree, args.output)
    else:
        print(f"Note: Visualization for {metadata['n_dimensions']}D data not yet implemented.")
        print("Analysis completed successfully. Raw persistence data available.")
    
    print(f"\nAnalysis complete. Check figures/{args.output}.png")
    print("\nChemical interpretation of sublevel topology:")
    print("- H0 features = Energy basins (local minima regions)")
    print("- H1 features = Transition cycles connecting basins")
    print("- Birth/death energies = Critical energies for topological changes")
    print("- Persistence = Energetic significance of topological features")


if __name__ == "__main__":
    main() 
