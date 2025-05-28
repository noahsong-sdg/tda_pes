\
import numpy as np
import gudhi as gd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import argparse
import os

def perform_tda_on_pes(angles, rel_energies, output_prefix="pentane"):
    """
    Performs sublevel set filtration on the PES data and computes homology.
    Tracks generators of 0-dimensional and 1-dimensional homology.
    Saves persistence diagram and barcode plots.
    """
    print(f"\n{'='*60}")
    print("TOPOLOGICAL DATA ANALYSIS OF PES")
    print(f"{'='*60}")

    if gd is None:
        print("Gudhi library not available. Skipping TDA.")
        return [], []
        
    if not angles.size or not rel_energies.size:
        print("No data to analyze.")
        return [], []

    st = gd.SimplexTree()
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_energies = rel_energies[sorted_indices]

    print("\\nAdding vertices to simplex tree...")
    for i in range(len(sorted_angles)):
        st.insert([i], filtration=sorted_energies[i])

    print("\\nAdding edges to simplex tree...")
    for i in range(len(sorted_angles) - 1):
        filtration_val = max(sorted_energies[i], sorted_energies[i+1])
        st.insert([i, i+1], filtration=filtration_val)
    
    if len(sorted_angles) > 1: # Periodic edge for 0-360 scan
        filtration_val_periodic_edge = max(sorted_energies[-1], sorted_energies[0])
        st.insert([len(sorted_angles)-1, 0], filtration=filtration_val_periodic_edge)

    print(f"\\nSimplex tree created with {st.num_vertices()} vertices and {st.num_simplices()} simplices.")
    st.set_dimension(1) 

    print("Computing persistence...")
    st.compute_persistence(homology_coeff_field=2, min_persistence=0.01) # min_persistence can be tuned
    all_persistence_pairs = st.persistence_pairs()

    print("\\nPersistence Details (H0 and H1):")
    generators_info = []
    diag_for_plotting = []

    print("\\n--- H0 (Connected Components) ---")
    h0_found = False
    print("\\n--- H1 (Cycles/Loops) ---")
    h1_found = False

    for birth_simplex_handle, death_simplex_handle in all_persistence_pairs:
        if birth_simplex_handle is None: continue

        dim = st.simplex_dimension(birth_simplex_handle)
        birth_val = st.filtration(birth_simplex_handle)
        death_val = float('inf')
        if death_simplex_handle is not None:
            death_val = st.filtration(death_simplex_handle)
        
        persistence_val = death_val - birth_val if death_val != float('inf') else float('inf')

        if dim not in [0, 1]: continue
        diag_for_plotting.append((dim, (birth_val, death_val)))

        if dim == 0:
            h0_found = True
            birth_vertex_indices = st.simplex_vertices(birth_simplex_handle)
            if not birth_vertex_indices: continue
            generator_vertex_index = birth_vertex_indices[0]
            generator_angle = sorted_angles[generator_vertex_index]
            print(f"  Birth: {birth_val:.2f} (Angle: {generator_angle:.1f}°, E: {sorted_energies[generator_vertex_index]:.2f}) | Death: {death_val:.2f} | Pers: {persistence_val:.2f}")
            generators_info.append({
                'dim': 0, 'birth': birth_val, 'death': death_val, 'persistence': persistence_val,
                'angle': generator_angle, 'energy': sorted_energies[generator_vertex_index]
            })
        elif dim == 1:
            h1_found = True
            birth_edge_indices = st.simplex_vertices(birth_simplex_handle)
            if len(birth_edge_indices) < 2: continue
            v1_idx, v2_idx = birth_edge_indices[0], birth_edge_indices[1]
            edge_angles = (sorted_angles[v1_idx], sorted_angles[v2_idx])
            edge_energies = (sorted_energies[v1_idx], sorted_energies[v2_idx])
            print(f"  Birth: {birth_val:.2f} (Edge: {edge_angles[0]:.1f}° [E={edge_energies[0]:.2f}] - {edge_angles[1]:.1f}° [E={edge_energies[1]:.2f}]) | Death: {death_val:.2f} | Pers: {persistence_val:.2f}")
            generators_info.append({
                'dim': 1, 'birth': birth_val, 'death': death_val, 'persistence': persistence_val,
                'edge_angles': edge_angles, 'edge_energies': edge_energies
            })

    if not h0_found: print("  No H0 features found with current min_persistence.")
    if not h1_found: print("  No H1 features found with current min_persistence.")

    if diag_for_plotting:
        os.makedirs('figures', exist_ok=True) # Ensure figures directory exists
        plt.figure(figsize=(8,8))
        gd.plot_persistence_diagram(diag_for_plotting)
        plt.title(f"Persistence Diagram for {output_prefix} PES")
        plt.tight_layout()
        diag_filename = f'figures/{output_prefix}_persistence_diagram.png'
        plt.savefig(diag_filename, dpi=300)
        print(f"\\nPersistence diagram saved to {diag_filename}")
        plt.close()

        plt.figure(figsize=(10, 4))
        gd.plot_persistence_barcode(diag_for_plotting)
        plt.title(f"Persistence Barcode for {output_prefix} PES")
        plt.tight_layout()
        barcode_filename = f'figures/{output_prefix}_persistence_barcode.png'
        plt.savefig(barcode_filename, dpi=300)
        print(f"Persistence barcode saved to {barcode_filename}")
        plt.close()
    else:
        print("No persistence pairs to plot.")
        
    return generators_info, diag_for_plotting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform TDA on PES data from a file.")
    parser.add_argument('pes_data_file', type=str, 
                        help="Path to the PES data file (e.g., 'data/pentane_pes.dat').")
    parser.add_argument('--output-prefix', type=str, default='pentane',
                        help="Prefix for output plot filenames (plots saved in figures/ e.g., 'pentane').")

    args = parser.parse_args()

    # Ensure figures directory exists for outputs
    os.makedirs('figures', exist_ok=True)

    if not os.path.exists(args.pes_data_file):
        print(f"Error: PES data file '{args.pes_data_file}' not found.")
        exit(1)

    print(f"Loading PES data from {args.pes_data_file}...")
    try:
        data = np.loadtxt(args.pes_data_file, skiprows=1) # Assumes header line
        angles, rel_energies = data[:, 0], data[:, 1]
        print("PES data loaded successfully.")
    except Exception as e:
        print(f"Error loading PES data file: {e}")
        exit(1)

    perform_tda_on_pes(angles, rel_energies, args.output_prefix)
    print("\nTDA processing complete.")
