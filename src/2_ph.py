import numpy as np
import os
import argparse

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi as gd

def load_pes_data(filename):
    """Load PES data from the specified file."""
    if not os.path.exists(filename):
        print(f"Error: PES data file '{filename}' not found.")
        print("Please run 1_pes.py first to generate the PES data.")
        return None, None
    
    print(f"Loading PES data from {filename}...")
    data = np.loadtxt(filename)
    angles = data[:, 0]
    energies = data[:, 1]
    
    # Calculate relative energies
    min_energy = np.min(energies)
    rel_energies = energies - min_energy
    
    print(f"Loaded {len(angles)} data points")
    print(f"Energy range: {np.min(rel_energies):.6f} to {np.max(rel_energies):.6f} Hartree")
    
    return angles, rel_energies

def perform_tda_on_pes(angles, rel_energies):
    """
    Perform sublevel set persistent homology on the PES of pentane using Gudhi's CubicalComplex.
    """
    print("="*60)
    print("TOPOLOGICAL DATA ANALYSIS OF PES (Sublevel Filtration)")
    print("="*60)

    if gd is None:
        print("Gudhi library not available.")
        return []

    if not angles.size or not rel_energies.size:
        print("No data provided.")
        return []

    # Sort data
    sorted_indices = np.argsort(angles)
    angles = angles[sorted_indices]
    energies = rel_energies[sorted_indices]
    n = len(energies)

    print(f"Analyzing {n} data points from {angles[0]:.1f}° to {angles[-1]:.1f}°")
    print(f"Energy range: {np.min(energies):.6f} to {np.max(energies):.6f} Hartree")
    
    # Method 1: Use CubicalComplex for proper sublevel set filtration
    print("\n--- METHOD 1: Cubical Complex Sublevel Filtration ---")
    
    # Create a 1D cubical complex (line) with energy values
    # We need to make the data periodic by duplicating the first point at the end
    periodic_energies = np.append(energies, energies[0])
    
    # Create cubical complex - Gudhi expects the function values directly
    cubical_complex = gd.CubicalComplex(dimensions=[len(periodic_energies)], 
                                       top_dimensional_cells=periodic_energies)
    
    print(f"Cubical complex created with {len(periodic_energies)} cells")
    
    # Compute persistence
    persistence_cubical = cubical_complex.persistence()
    
    print(f"Cubical persistence computed: {len(persistence_cubical)} pairs")
    
    # Method 2: Manual circular graph with proper sublevel filtration
    print("\n--- METHOD 2: Circular Graph with SimplexTree ---")
    
    st = gd.SimplexTree()
    
    # Add vertices with energy as filtration value (sublevel set)
    for i, energy in enumerate(energies):
        st.insert([i], filtration=energy)
    
    # Add edges - filtration is max of endpoint energies (for sublevel sets)
    for i in range(n-1):
        edge_filt = max(energies[i], energies[i+1])
        st.insert([i, i+1], filtration=edge_filt)
    
    # Add periodic edge to close the circle
    periodic_filt = max(energies[0], energies[-1])
    st.insert([0, n-1], filtration=periodic_filt)
    
    # For H1 features, we need 2-simplices (triangles)
    # Add triangles connecting nearby vertices to capture cycles
    print("Adding triangles to capture H1 features...")
    triangle_count = 0
    
    # Add triangles for consecutive triplets
    for i in range(n):
        j = (i + 1) % n
        k = (i + 2) % n
        triangle_filt = max(energies[i], energies[j], energies[k])
        st.insert([i, j, k], filtration=triangle_filt)
        triangle_count += 1
    
    # Add some longer-range triangles to better capture the topology
    step = max(1, n // 12)  # Create ~12 additional triangles
    for i in range(0, n, step):
        j = (i + step) % n
        k = (i + 2*step) % n
        triangle_filt = max(energies[i], energies[j], energies[k])
        st.insert([i, j, k], filtration=triangle_filt)
        triangle_count += 1
    
    print(f"Added {triangle_count} triangles")
    
    # Initialize filtration and compute persistence
    st.initialize_filtration()
    persistence_graph = st.persistence(homology_coeff_field=2, min_persistence=0)
    
    print(f"Graph complex: {st.num_vertices()} vertices, {st.num_simplices()} simplices")
    print(f"Graph persistence computed: {len(persistence_graph)} pairs")
    
    # Compare results and choose the better one
    methods = [
        ("Cubical Complex", persistence_cubical),
        ("Circular Graph", persistence_graph)
    ]
    
    for method_name, persistence in methods:
        print(f"\n--- ANALYZING {method_name.upper()} ---")
        h0_count = h1_count = h2_count = 0
        
        for dim, (birth, death) in persistence:
            if dim == 0:
                h0_count += 1
            elif dim == 1:
                h1_count += 1
            elif dim == 2:
                h2_count += 1
        
        print(f"Found H0: {h0_count}, H1: {h1_count}, H2: {h2_count}")
    
    # Use the method that found H1 features, or circular graph if both have same H1 count
    best_persistence = persistence_graph
    best_simplex_tree = st
    for method_name, persistence in methods:
        h1_count = sum(1 for dim, _ in persistence if dim == 1)
        if h1_count > 0:
            best_persistence = persistence
            if method_name == "Circular Graph":
                best_simplex_tree = st
            else:
                best_simplex_tree = None  # Cubical complex doesn't provide simplex tree
            print(f"\nUsing {method_name} results (found H1 features)")
            break
    else:
        print(f"\nUsing Circular Graph results (default)")
    
    # Analyze and return the best persistence results
    return analyze_persistence_results(best_persistence, energies, best_simplex_tree, angles)

def analyze_persistence_results(persistence, energies, simplex_tree=None, angles=None):
    """Analyze and plot persistence results with generator extraction."""
    diag_for_plotting = []
    generators_info = []

    print("\n--- PERSISTENCE GENERATORS ---")
    h0_count = 0
    h1_count = 0

    # Extract persistence pairs with generators if simplex tree is available
    for i, (dim, (birth, death)) in enumerate(persistence):
        if dim not in [0, 1]:
            continue  # Only interested in H0 and H1

        persistence_val = death - birth if death != float('inf') else float('inf')
        diag_for_plotting.append((dim, (birth, death)))
        
        # Store generator info
        generator_info = {
            'dimension': dim,
            'birth': birth,
            'death': death,
            'persistence': persistence_val,
            'index': i,
            'birth_simplex': None,
            'death_simplex': None
        }
        
        # Try to get generator information from simplex tree
        if simplex_tree is not None:
            try:
                # Get persistence pairs with simplices
                persistence_pairs = simplex_tree.persistence_pairs()
                if i < len(persistence_pairs):
                    birth_simplex, death_simplex = persistence_pairs[i]
                    generator_info['birth_simplex'] = birth_simplex
                    generator_info['death_simplex'] = death_simplex
            except:
                pass  # Continue without generator info if extraction fails
        
        generators_info.append(generator_info)

        if dim == 0:
            # H0: Connected component
            h0_count += 1
            print(f"[H0-{h0_count}] Component born at E={birth:.6f}, dies at E={death:.6f} → persistence = {persistence_val:.6f}")
            if generator_info['birth_simplex'] is not None:
                print(f"    Birth simplex: {generator_info['birth_simplex']}")
        elif dim == 1:
            # H1: Cycle
            h1_count += 1
            print(f"[H1-{h1_count}] Cycle born at E={birth:.6f}, dies at E={death:.6f} → persistence = {persistence_val:.6f}")
            if generator_info['birth_simplex'] is not None:
                print(f"    Birth simplex: {generator_info['birth_simplex']}")

    if h0_count == 0:
        print("No H0 generators found.")
    if h1_count == 0:
        print("No H1 generators found.")

    # Create comprehensive visualization
    create_comprehensive_visualization(angles, energies, generators_info, diag_for_plotting)

    print("="*60)
    return persistence

def create_comprehensive_visualization(angles, energies, generators_info, diag_for_plotting):
    """Create comprehensive visualization showing PES with generators and barcode diagram."""
    
    # Ensure figures directory exists
    os.makedirs("../figures", exist_ok=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Define colors for different homology dimensions
    h0_color = 'red'
    h1_color = 'blue'
    
    # Subplot 1: PES with highlighted generators
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot the basic PES
    ax1.plot(angles, energies, 'k-', linewidth=2, alpha=0.7, label='PES')
    ax1.scatter(angles, energies, c='gray', s=30, alpha=0.6, zorder=3)
    
    # Highlight generators with labels
    h0_generators = []
    h1_generators = []
    h1_generator_count = 0
    
    for gen_info in generators_info:
        if gen_info['birth_simplex'] is not None:
            birth_simplex = gen_info['birth_simplex']
            
            if gen_info['dimension'] == 0:
                # H0 generator: highlight the birth vertex
                if len(birth_simplex) == 1:
                    vertex_idx = birth_simplex[0]
                    if vertex_idx < len(angles):
                        ax1.scatter(angles[vertex_idx], energies[vertex_idx], 
                                  c=h0_color, s=100, marker='o', zorder=5, 
                                  edgecolors='black', linewidth=1)
                        h0_generators.append((angles[vertex_idx], energies[vertex_idx]))
                        
            elif gen_info['dimension'] == 1:
                # H1 generator: highlight the cycle with numbered labels
                h1_generator_count += 1
                if len(birth_simplex) >= 2:
                    # Extract vertices from the birth simplex (edge or triangle)
                    vertices = birth_simplex
                    max_energy_vertex = None
                    max_energy = -float('inf')
                    
                    for vertex_idx in vertices:
                        if vertex_idx < len(angles):
                            ax1.scatter(angles[vertex_idx], energies[vertex_idx], 
                                      c=h1_color, s=100, marker='s', zorder=5,
                                      edgecolors='black', linewidth=1)
                            
                            # Track vertex with highest energy for label placement
                            if energies[vertex_idx] > max_energy:
                                max_energy = energies[vertex_idx]
                                max_energy_vertex = vertex_idx
                    
                    # Add label to the highest energy vertex in this generator
                    if max_energy_vertex is not None:
                        ax1.annotate(f'H1-{h1_generator_count}', 
                                   xy=(angles[max_energy_vertex], energies[max_energy_vertex]),
                                   xytext=(8, 12), textcoords='offset points',
                                   fontsize=9, fontweight='bold', color=h1_color,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                           edgecolor=h1_color, alpha=0.9),
                                   arrowprops=dict(arrowstyle='->', color=h1_color, lw=1.2))
                    
                    h1_generators.append(vertices)
    
    ax1.set_xlabel('Dihedral Angle (degrees)')
    ax1.set_ylabel('Relative Energy (Hartree)')
    ax1.set_title('PES with Highlighted Generators')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add legend for generators
    if h0_generators or h1_generators:
        legend_elements = []
        if h0_generators:
            legend_elements.append(plt.scatter([], [], c=h0_color, s=100, marker='o', 
                                             edgecolors='black', linewidth=1, label='H0 Generators'))
        if h1_generators:
            legend_elements.append(plt.scatter([], [], c=h1_color, s=100, marker='s', 
                                             edgecolors='black', linewidth=1, label='H1 Generators'))
        ax1.legend(handles=legend_elements, loc='upper right')
    
    # Subplot 2: Persistence barcode with color coding
    ax2 = plt.subplot(2, 2, 2)
    
    # Create custom barcode plot
    y_pos = 0
    bar_height = 0.8
    
    # Separate H0 and H1 features
    h0_features = [(birth, death) for dim, (birth, death) in diag_for_plotting if dim == 0]
    h1_features = [(birth, death) for dim, (birth, death) in diag_for_plotting if dim == 1]
    
    # Plot H0 bars
    for i, (birth, death) in enumerate(h0_features):
        if death == float('inf'):
            death = max(energies) * 1.1  # Extend infinite bars
        ax2.barh(y_pos, death - birth, left=birth, height=bar_height, 
                color=h0_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.text(birth + (death - birth) / 2, y_pos, f'H0-{i+1}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
        y_pos += 1
    
    # Plot H1 bars
    for i, (birth, death) in enumerate(h1_features):
        if death == float('inf'):
            death = max(energies) * 1.1
        ax2.barh(y_pos, death - birth, left=birth, height=bar_height, 
                color=h1_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.text(birth + (death - birth) / 2, y_pos, f'H1-{i+1}', 
                ha='center', va='center', fontsize=8, fontweight='bold')
        y_pos += 1
    
    ax2.set_xlabel('Energy (Hartree)')
    ax2.set_ylabel('Features')
    ax2.set_title('Persistence Barcode')
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis labels
    y_labels = [f'H0-{i+1}' for i in range(len(h0_features))] + \
               [f'H1-{i+1}' for i in range(len(h1_features))]
    ax2.set_yticks(range(len(y_labels)))
    ax2.set_yticklabels(y_labels)
    
    # Subplot 3: Persistence diagram
    ax3 = plt.subplot(2, 2, 3)
    
    # Plot H0 points
    h0_births = [birth for birth, death in h0_features]
    h0_deaths = [death if death != float('inf') else max(energies) * 1.1 for birth, death in h0_features]
    if h0_births:
        ax3.scatter(h0_births, h0_deaths, c=h0_color, s=80, alpha=0.8, 
                   label=f'H0 ({len(h0_features)} features)', edgecolors='black')
    
    # Plot H1 points
    h1_births = [birth for birth, death in h1_features]
    h1_deaths = [death if death != float('inf') else max(energies) * 1.1 for birth, death in h1_features]
    if h1_births:
        ax3.scatter(h1_births, h1_deaths, c=h1_color, s=80, alpha=0.8, 
                   label=f'H1 ({len(h1_features)} features)', edgecolors='black')
    
    # Plot diagonal line
    max_val = max(max(energies), max(h0_deaths + h1_deaths)) if (h0_deaths + h1_deaths) else max(energies)
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Birth = Death')
    
    ax3.set_xlabel('Birth (Hartree)')
    ax3.set_ylabel('Death (Hartree)')
    ax3.set_title('Persistence Diagram')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Subplot 4: Generator summary table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Create summary text
    summary_text = "GENERATOR SUMMARY\n" + "="*30 + "\n\n"
    
    # H0 summary
    h0_count = len([g for g in generators_info if g['dimension'] == 0])
    summary_text += f"H0 (Connected Components): {h0_count}\n"
    for i, gen_info in enumerate([g for g in generators_info if g['dimension'] == 0]):
        summary_text += f"  H0-{i+1}: Birth={gen_info['birth']:.4f}, "
        summary_text += f"Death={gen_info['death']:.4f}, "
        summary_text += f"Persistence={gen_info['persistence']:.4f}\n"
    
    summary_text += "\n"
    
    # H1 summary
    h1_count = len([g for g in generators_info if g['dimension'] == 1])
    summary_text += f"H1 (Cycles): {h1_count}\n"
    for i, gen_info in enumerate([g for g in generators_info if g['dimension'] == 1]):
        summary_text += f"  H1-{i+1}: Birth={gen_info['birth']:.4f}, "
        summary_text += f"Death={gen_info['death']:.4f}, "
        summary_text += f"Persistence={gen_info['persistence']:.4f}\n"
    
    # Add interpretation
    summary_text += "\n" + "INTERPRETATION\n" + "="*30 + "\n"
    summary_text += f"• Total features found: {len(generators_info)}\n"
    summary_text += f"• H0 features represent connected components\n"
    summary_text += f"• H1 features represent cycles in the PES\n"
    summary_text += f"• Persistence = Death - Birth (longer bars = more significant)\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("../figures/pentane_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis to ../figures/pentane_comprehensive_analysis.png")
    
    # Also save individual plots
    try:
        plt.figure(figsize=(8, 8))
        gd.plot_persistence_diagram(diag_for_plotting)
        plt.title("Persistence Diagram")
        plt.tight_layout()
        plt.savefig("../figures/pentane_persistence_diagram.png", dpi=300)
        print("Saved persistence diagram to ../figures/pentane_persistence_diagram.png")

        plt.figure(figsize=(10, 4))
        gd.plot_persistence_barcode(diag_for_plotting)
        plt.title("Persistence Barcode")
        plt.tight_layout()
        plt.savefig("../figures/pentane_persistence_barcode.png", dpi=300)
        print("Saved persistence barcode to ../figures/pentane_persistence_barcode.png")
    except Exception as e:
        print(f"Standard plotting error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Perform Topological Data Analysis on Pentane PES")
    parser.add_argument("--data-file", default="../data/pentane_pes.dat", 
                       help="Path to PES data file (default: ../data/pentane_pes.dat)")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PENTANE PES TOPOLOGICAL DATA ANALYSIS")
    print("="*60)
    
    # Load PES data
    angles, rel_energies = load_pes_data(args.data_file)
    if angles is None or rel_energies is None:
        return
    
    # Perform TDA analysis
    persistence = perform_tda_on_pes(angles, rel_energies)
    
    if persistence:
        print("\nAnalysis completed successfully!")
        print(f"Found {len(persistence)} persistence pairs")
    else:
        print("\nAnalysis failed or no persistence features found.")

if __name__ == "__main__":
    main()
