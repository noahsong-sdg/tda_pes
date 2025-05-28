\
import numpy as np
import pyscf
from pyscf import gto, scf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Define Z-matrix constants
R_CC = 1.54  # Angstrom
R_CH = 1.09  # Angstrom
A_CCC = 109.47  # degrees
A_CCH = 109.47  # degrees
D_TRANS = 180.0  # degrees for trans C-C-C-C
D_HCH_OFFSET = 120.0  # degrees for staggering H's

def create_pentane_geometry(dihedral_angle):
    """
    Create pentane geometry using a Z-matrix with a specified C2-C3-C4-C5 dihedral angle.
    The dihedral_angle corresponds to the rotation around the C3-C4 bond.
    """
    zmatrix_string = f"""
C
C   1  {R_CC}
H   1  {R_CH}   2  {A_CCH}
H   1  {R_CH}   2  {A_CCH}   3  {D_HCH_OFFSET}
H   1  {R_CH}   2  {A_CCH}   3 {-D_HCH_OFFSET}
C   2  {R_CC}   1  {A_CCC}   3  {D_TRANS}
H   2  {R_CH}   6  {A_CCH}   1  {D_HCH_OFFSET}
H   2  {R_CH}   6  {A_CCH}   1 {-D_HCH_OFFSET}
C   6  {R_CC}   2  {A_CCC}   1  {D_TRANS}
H   6  {R_CH}   9  {A_CCH}   2  {D_HCH_OFFSET}
H   6  {R_CH}   9  {A_CCH}   2 {-D_HCH_OFFSET}
C   9  {R_CC}   6  {A_CCC}   2  {dihedral_angle}
H   9  {R_CH}  12  {A_CCH}   6  {D_HCH_OFFSET}
H   9  {R_CH}  12  {A_CCH}   6 {-D_HCH_OFFSET}
H  12  {R_CH}   9  {A_CCH}   6  {D_TRANS}
H  12  {R_CH}   9  {A_CCH}  15  {D_HCH_OFFSET}
H  12  {R_CH}   9  {A_CCH}  15 {-D_HCH_OFFSET}
"""
    return zmatrix_string

def print_coordinates(coords, angle, energy):
    """Print atomic coordinates in a formatted table"""
    print(f"\n{'='*60}")
    print(f"MOLECULAR CONFIGURATION - Dihedral Angle: {angle:.1f}°")
    print(f"Energy: {energy:.6f} Hartree")
    print(f"{'='*60}")
    print(f"{'Atom':<6} {'X (Å)':<10} {'Y (Å)':<10} {'Z (Å)':<10}")
    print(f"{'-'*60}")
    
    for i, (element, (x, y, z)) in enumerate(coords):
        print(f"{element}{i+1:<5} {x:<10.4f} {y:<10.4f} {z:<10.4f}")
    
    print(f"{'-'*60}")

def visualize_molecule(coords, angle, energy, save_plot=True):
    """Visualize the molecular structure in 3D"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    carbons = []
    hydrogens = []
    
    for atom in coords:
        if atom[0] == 'C':
            carbons.append(atom[1])
        else:
            hydrogens.append(atom[1])
    
    carbons_np = np.array(carbons) if carbons else np.array([]).reshape(0,3)
    hydrogens_np = np.array(hydrogens) if hydrogens else np.array([]).reshape(0,3)

    if carbons_np.size > 0:
        ax.scatter(carbons_np[:, 0], carbons_np[:, 1], carbons_np[:, 2], 
                  c='black', s=200, alpha=0.8, label='Carbon')
    
    if hydrogens_np.size > 0:
        ax.scatter(hydrogens_np[:, 0], hydrogens_np[:, 1], hydrogens_np[:, 2], 
                  c='lightgray', s=100, alpha=0.6, label='Hydrogen')
    
    if carbons_np.shape[0] >= 2:
        for i in range(len(carbons_np) - 1):
            ax.plot([carbons_np[i, 0], carbons_np[i+1, 0]], 
                   [carbons_np[i, 1], carbons_np[i+1, 1]], 
                   [carbons_np[i, 2], carbons_np[i+1, 2]], 
                   'k-', linewidth=2, alpha=0.7)
    
    if carbons_np.size > 0 and hydrogens_np.size > 0:
        for carbon_coord in carbons_np:
            for hydrogen_coord in hydrogens_np:
                distance = np.linalg.norm(carbon_coord - hydrogen_coord)
                if distance < 1.5:  # Typical C-H bond length threshold
                    ax.plot([carbon_coord[0], hydrogen_coord[0]], 
                           [carbon_coord[1], hydrogen_coord[1]], 
                           [carbon_coord[2], hydrogen_coord[2]], 
                           'gray', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.legend()
    ax.set_title(f'Pentane Configuration\nDihedral Angle: {angle:.1f}°\nEnergy: {energy:.6f} Hartree')
    ax.grid(True, alpha=0.3)
    
    if save_plot:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/pentane_structure_{angle:.0f}deg.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    return fig, ax

def calculate_energy(dihedral_angle, visualize=False, print_coords_flag=False): # Renamed print_coords to avoid conflict
    """Calculate energy for given dihedral angle using Z-matrix and fast HF/STO-3G"""
    zmatrix_str = create_pentane_geometry(dihedral_angle)
    
    mol = gto.Mole()
    mol.atom = zmatrix_str
    mol.basis = 'sto-3g'
    mol.charge = 0
    mol.spin = 0
    mol.build(verbose=0)
    
    mf = scf.RHF(mol)
    mf.verbose = 0
    energy = mf.kernel()
    
    if print_coords_flag or visualize:
        atom_symbols = mol.elements
        xyz_coords = mol.atom_coords(unit='Angstrom')
        formatted_coords = [[atom_symbols[i], tuple(xyz_coords[i])] for i in range(mol.natm)]

        if print_coords_flag:
            print_coordinates(formatted_coords, dihedral_angle, energy) 
    
        if visualize:
            visualize_molecule(formatted_coords, dihedral_angle, energy, save_plot=True)
    return energy

def create_pes(visualize_configs=False, save_individual_plots=False):
    """Create potential energy surface by scanning dihedral angles"""
    angles = np.linspace(0, 360, 37)  # 37 points for 0 to 360 inclusive, every 10 deg.
    energies = []
    
    print("Calculating PES...")
    for i, angle in enumerate(angles):
        print(f"Angle {angle:.1f}° ({i+1}/{len(angles)})")
        show_vis = save_individual_plots and (i % 5 == 0)
        show_coords = save_individual_plots and (i % 5 == 0) 
        
        energy = calculate_energy(angle, visualize=show_vis, print_coords_flag=show_coords)
        energies.append(energy)
    
    energies = np.array(energies)
    rel_energies = (energies - np.min(energies)) * 627.509  # Hartree to kcal/mol
    
    fig_comp, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(angles, rel_energies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Dihedral Angle (degrees)')
    ax1.set_ylabel('Relative Energy (kcal/mol)')
    ax1.set_title('Potential Energy Surface of Pentane')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 360)
    min_idx = np.argmin(rel_energies)
    ax1.scatter(angles[min_idx], rel_energies[min_idx], color='red', s=100, 
               label=f'Minimum at {angles[min_idx]:.1f}°', zorder=5)
    ax1.legend()
    
    ax2.hist(rel_energies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Relative Energy (kcal/mol)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Energy Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/pentane_pes_comprehensive.png', dpi=300)
    plt.close(fig_comp)
    
    fig_simple = plt.figure(figsize=(10, 6))
    plt.plot(angles, rel_energies, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Dihedral Angle (degrees)')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.title('Potential Energy Surface of Pentane')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 360)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/pentane_pes.png', dpi=300)
    plt.close(fig_simple)
    
    os.makedirs('data', exist_ok=True)
    np.savetxt('data/pentane_pes.dat', np.column_stack([angles, rel_energies]), 
               header='Angle(deg) Energy(kcal/mol)')
    
    print(f"\\n{'='*60}")
    print("ENERGY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Minimum energy: {np.min(rel_energies):.3f} kcal/mol at {angles[min_idx]:.1f}°")
    print(f"Maximum energy: {np.max(rel_energies):.3f} kcal/mol at {angles[np.argmax(rel_energies)]:.1f}°")
    print(f"Energy range: {np.max(rel_energies) - np.min(rel_energies):.3f} kcal/mol")
    print(f"Average energy: {np.mean(rel_energies):.3f} kcal/mol")
    print(f"Standard deviation: {np.std(rel_energies):.3f} kcal/mol")
    
    if visualize_configs:
        print(f"\nVisualizing minimum energy configuration at {angles[min_idx]:.1f}°...")
        calculate_energy(angles[min_idx], visualize=True, print_coords_flag=True)
    
    return angles, rel_energies

def demonstrate_single_configuration(angle=180):
    """Demonstrate atomic configuration for a single angle"""
    print(f"Demonstrating pentane configuration at {angle}° dihedral angle")
    energy = calculate_energy(angle, visualize=True, print_coords_flag=True)
    print(f"Calculated energy for {angle}°: {energy:.6f} Hartree")
    return energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pentane PES Calculator. Generates 'data/pentane_pes.dat'.")
    parser.add_argument('angle', type=float, nargs='?', default=None,
                        help="Calculate and visualize for a single dihedral angle (degrees).")
    parser.add_argument('--recalculate', action='store_true',
                        help="Force recalculation of PES data even if 'data/pentane_pes.dat' exists.")
    parser.add_argument('--visualize-min-config', action='store_true',
                        help="Visualize the minimum energy configuration after PES scan (or from loaded data).")
    parser.add_argument('--save-step-plots', action='store_true',
                        help="Save individual molecular structure images (in figures/) during PES scan.")
    
    args = parser.parse_args()
    pes_data_file = 'data/pentane_pes.dat'

    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    if args.angle is not None:
        demonstrate_single_configuration(args.angle)
    else:
        if not args.recalculate and os.path.exists(pes_data_file):
            print(f"Loading PES data from {pes_data_file}...")
            data = np.loadtxt(pes_data_file, skiprows=1)
            angles, rel_energies = data[:, 0], data[:, 1]
            print("PES data loaded.")
            
            min_idx = np.argmin(rel_energies)
            print(f"Minimum energy from loaded data: {np.min(rel_energies):.3f} kcal/mol at {angles[min_idx]:.1f}°")

            if args.visualize_min_config:
                 print(f"\nVisualizing minimum energy configuration at {angles[min_idx]:.1f}° from loaded data...")
                 calculate_energy(angles[min_idx], visualize=True, print_coords_flag=True)
        else:
            if args.recalculate and os.path.exists(pes_data_file):
                print(f"Recalculating PES data ('--recalculate' specified)...")
            elif not os.path.exists(pes_data_file):
                 print(f"PES data file ('{pes_data_file}') not found. Calculating PES...")
            
            angles, rel_energies = create_pes(
                visualize_configs=args.visualize_min_config, 
                save_individual_plots=args.save_step_plots
            )
        
        print(f"\\nPES generation/loading complete. Data is in '{pes_data_file}'.")
        print("To perform TDA, run 2_ph.py using this data file (e.g., python src/2_ph.py data/pentane_pes.dat).")
