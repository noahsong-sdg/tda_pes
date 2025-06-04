
import numpy as np
import pyscf
from pyscf import gto, scf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

import re

def read_zmatrix_template(filename):
    """
    Read Z-matrix template from file.
    The file should contain a Z-matrix with variable parameters marked as {variable_name}.
    Returns the template string and a list of variable names found.
    """
    with open(filename, 'r') as f:
        template = f.read().strip()
    
    # Find all variables in curly braces
    variables = re.findall(r'\{([^}]+)\}', template)
    unique_variables = list(set(variables))
    
    print(f"Found variables in Z-matrix: {unique_variables}")
    return template, unique_variables

def create_geometry_from_template(template, variable_values):
    """
    Create geometry by substituting variables in the Z-matrix template.
    
    Args:
        template (str): Z-matrix template with {variable_name} placeholders
        variable_values (dict): Dictionary mapping variable names to values
    
    Returns:
        str: Z-matrix string with variables substituted
    """
    zmatrix_string = template
    for var_name, value in variable_values.items():
        zmatrix_string = zmatrix_string.replace(f'{{{var_name}}}', str(value))
    
    return zmatrix_string

def get_default_variable_values(variables):
    """
    Get default values for variables. This can be customized based on variable names.
    """
    defaults = {}
    for var in variables:
        var_lower = var.lower()
        if 'dihedral' in var_lower or 'torsion' in var_lower:
            defaults[var] = 0.0  # Default dihedral angle
        elif 'angle' in var_lower:
            defaults[var] = 109.47  # Default tetrahedral angle
        elif 'bond' in var_lower or 'r_' in var_lower:
            if 'cc' in var_lower:
                defaults[var] = 1.54  # C-C bond length
            elif 'ch' in var_lower:
                defaults[var] = 1.09  # C-H bond length
            else:
                defaults[var] = 1.5  # Generic bond length
        else:
            defaults[var] = 0.0  # Default for unknown variables
    
    return defaults

def print_coordinates(coords, variable_values, energy):
    """Print atomic coordinates in a formatted table"""
    print(f"\n{'='*60}")
    print(f"MOLECULAR CONFIGURATION - Variables: {variable_values}")
    print(f"Energy: {energy:.6f} Hartree")
    print(f"{'='*60}")
    print(f"{'Atom':<6} {'X (Å)':<10} {'Y (Å)':<10} {'Z (Å)':<10}")
    print(f"{'-'*60}")
    
    for i, (element, (x, y, z)) in enumerate(coords):
        print(f"{element}{i+1:<5} {x:<10.4f} {y:<10.4f} {z:<10.4f}")
    
    print(f"{'-'*60}")

def visualize_molecule(coords, variable_values, energy, molecule_name="molecule", save_plot=True):
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
    
    # Create a more generic title
    var_str = ", ".join([f"{k}: {v:.1f}" for k, v in variable_values.items()])
    ax.set_title(f'{molecule_name.capitalize()} Configuration\n{var_str}\nEnergy: {energy:.6f} Hartree')
    ax.grid(True, alpha=0.3)
    
    if save_plot:
        os.makedirs('figures', exist_ok=True)
        # Create filename from variable values
        var_filename = "_".join([f"{k}{v:.0f}" for k, v in variable_values.items()])
        plt.savefig(f'figures/{molecule_name}_structure_{var_filename}.png', dpi=300, bbox_inches='tight')
    plt.close(fig) # Close the figure to free memory
    return fig, ax
    return fig, ax

def calculate_energy(variable_values, zmatrix_template, molecule_name="molecule", visualize=False, print_coords_flag=False):
    """Calculate energy for given variable values using Z-matrix template and fast HF/STO-3G"""
    zmatrix_str = create_geometry_from_template(zmatrix_template, variable_values)
    
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
            print_coordinates(formatted_coords, variable_values, energy) 
    
        if visualize:
            visualize_molecule(formatted_coords, variable_values, energy, molecule_name, save_plot=True)
    return energy

def create_pes(zmatrix_file, scan_variable=None, scan_range=None, num_points=37, molecule_name=None, 
               visualize_configs=False, save_individual_plots=False):
    """
    Create potential energy surface by scanning one variable parameter.
    
    Args:
        zmatrix_file (str): Path to Z-matrix template file
        scan_variable (str): Name of the variable to scan (if None, uses first variable)
        scan_range (tuple): Range to scan as (min, max) (default: (0, 360) for angles)
        num_points (int): Number of points to calculate
        molecule_name (str): Name of the molecule for output files
        visualize_configs (bool): Whether to visualize configurations
        save_individual_plots (bool): Whether to save individual structure plots
    """
    # Read the Z-matrix template
    zmatrix_template, variables = read_zmatrix_template(zmatrix_file)
    
    if not variables:
        raise ValueError("No variables found in Z-matrix template. Variables should be marked as {variable_name}")
    
    # Determine scan variable
    if scan_variable is None:
        scan_variable = variables[0]
        print(f"No scan variable specified. Using first variable: {scan_variable}")
    elif scan_variable not in variables:
        raise ValueError(f"Scan variable '{scan_variable}' not found in template. Available: {variables}")
    
    # Set default scan range based on variable name
    if scan_range is None:
        var_lower = scan_variable.lower()
        if 'dihedral' in var_lower or 'torsion' in var_lower:
            scan_range = (0, 360)
        elif 'angle' in var_lower:
            scan_range = (90, 150)
        else:
            scan_range = (0, 10)  # Generic range
        print(f"No scan range specified. Using default range: {scan_range}")
    
    # Get default values for all variables
    default_values = get_default_variable_values(variables)
    
    # Determine molecule name
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
    
    # Create scan values
    scan_values = np.linspace(scan_range[0], scan_range[1], num_points)
    energies = []
    
    print(f"Calculating PES for {molecule_name}...")
    print(f"Scanning {scan_variable} from {scan_range[0]} to {scan_range[1]} with {num_points} points")
    
    for i, value in enumerate(scan_values):
        print(f"{scan_variable} = {value:.1f} ({i+1}/{len(scan_values)})")
        
        # Set up variable values for this calculation
        variable_values = default_values.copy()
        variable_values[scan_variable] = value
        
        show_vis = save_individual_plots and (i % 5 == 0)
        show_coords = save_individual_plots and (i % 5 == 0) 
        
        energy = calculate_energy(variable_values, zmatrix_template, molecule_name, 
                                visualize=show_vis, print_coords_flag=show_coords)
        energies.append(energy)
    
    energies = np.array(energies)
    rel_energies = (energies - np.min(energies)) * 627.509  # Hartree to kcal/mol
    
    # Create comprehensive plot
    fig_comp, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(scan_values, rel_energies, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel(f'{scan_variable}')
    ax1.set_ylabel('Relative Energy (kcal/mol)')
    ax1.set_title(f'Potential Energy Surface of {molecule_name.capitalize()}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(scan_range[0], scan_range[1])
    min_idx = np.argmin(rel_energies)
    ax1.scatter(scan_values[min_idx], rel_energies[min_idx], color='red', s=100, 
               label=f'Minimum at {scan_values[min_idx]:.1f}', zorder=5)
    ax1.legend()
    
    ax2.hist(rel_energies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Relative Energy (kcal/mol)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Energy Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{molecule_name}_pes_comprehensive.png', dpi=300)
    plt.close(fig_comp)
    
    # Create simple plot
    fig_simple = plt.figure(figsize=(10, 6))
    plt.plot(scan_values, rel_energies, 'b-o', linewidth=2, markersize=6)
    plt.xlabel(f'{scan_variable}')
    plt.ylabel('Relative Energy (kcal/mol)')
    plt.title(f'Potential Energy Surface of {molecule_name.capitalize()}')
    plt.grid(True, alpha=0.3)
    plt.xlim(scan_range[0], scan_range[1])
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{molecule_name}_pes.png', dpi=300)
    plt.close(fig_simple)
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.savetxt(f'data/{molecule_name}_pes.dat', np.column_stack([scan_values, rel_energies]), 
               header=f'{scan_variable} Energy(kcal/mol)')
    
    print(f"\\n{'='*60}")
    print("ENERGY ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Minimum energy: {np.min(rel_energies):.3f} kcal/mol at {scan_variable} = {scan_values[min_idx]:.1f}")
    print(f"Maximum energy: {np.max(rel_energies):.3f} kcal/mol at {scan_variable} = {scan_values[np.argmax(rel_energies)]:.1f}")
    print(f"Energy range: {np.max(rel_energies) - np.min(rel_energies):.3f} kcal/mol")
    print(f"Average energy: {np.mean(rel_energies):.3f} kcal/mol")
    print(f"Standard deviation: {np.std(rel_energies):.3f} kcal/mol")
    
    if visualize_configs:
        print(f"\nVisualizing minimum energy configuration at {scan_variable} = {scan_values[min_idx]:.1f}...")
        min_variable_values = default_values.copy()
        min_variable_values[scan_variable] = scan_values[min_idx]
        calculate_energy(min_variable_values, zmatrix_template, molecule_name, 
                        visualize=True, print_coords_flag=True)
    
    return scan_values, rel_energies

def demonstrate_single_configuration(zmatrix_file, variable_values=None, molecule_name=None):
    """Demonstrate atomic configuration for specific variable values"""
    zmatrix_template, variables = read_zmatrix_template(zmatrix_file)
    
    if variable_values is None:
        variable_values = get_default_variable_values(variables)
    
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
    
    print(f"Demonstrating {molecule_name} configuration with variables: {variable_values}")
    energy = calculate_energy(variable_values, zmatrix_template, molecule_name, 
                            visualize=True, print_coords_flag=True)
    print(f"Calculated energy: {energy:.6f} Hartree")
    return energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General PES Calculator for molecular systems from Z-matrix files.")
    parser.add_argument('zmatrix_file', type=str,
                        help="Path to Z-matrix template file with variables marked as {variable_name}")
    parser.add_argument('--scan-variable', type=str, default=None,
                        help="Variable to scan (if not specified, uses first variable found)")
    parser.add_argument('--scan-range', type=float, nargs=2, default=None,
                        help="Range to scan as min max (default: 0 360 for angles)")
    parser.add_argument('--num-points', type=int, default=37,
                        help="Number of points to calculate (default: 37)")
    parser.add_argument('--molecule-name', type=str, default=None,
                        help="Name of the molecule for output files (default: filename without extension)")
    parser.add_argument('--single-point', action='store_true',
                        help="Calculate and visualize a single configuration with default values")
    parser.add_argument('--recalculate', action='store_true',
                        help="Force recalculation of PES data even if output file exists")
    parser.add_argument('--visualize-min-config', action='store_true',
                        help="Visualize the minimum energy configuration after PES scan")
    parser.add_argument('--save-step-plots', action='store_true',
                        help="Save individual molecular structure images during PES scan")
    
    args = parser.parse_args()
    
    # Check if Z-matrix file exists
    if not os.path.exists(args.zmatrix_file):
        print(f"Error: Z-matrix file '{args.zmatrix_file}' not found.")
        exit(1)
    
    # Determine molecule name
    molecule_name = args.molecule_name
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(args.zmatrix_file))[0]
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    if args.single_point:
        demonstrate_single_configuration(args.zmatrix_file, molecule_name=molecule_name)
    else:
        pes_data_file = f'data/{molecule_name}_pes.dat'
        
        if not args.recalculate and os.path.exists(pes_data_file):
            print(f"Loading PES data from {pes_data_file}...")
            data = np.loadtxt(pes_data_file, skiprows=1)
            scan_values, rel_energies = data[:, 0], data[:, 1]
            print("PES data loaded.")
            
            min_idx = np.argmin(rel_energies)
            print(f"Minimum energy from loaded data: {np.min(rel_energies):.3f} kcal/mol at {scan_values[min_idx]:.1f}")

            if args.visualize_min_config:
                # Need to reconstruct the scan variable and template for visualization
                zmatrix_template, variables = read_zmatrix_template(args.zmatrix_file)
                scan_variable = args.scan_variable if args.scan_variable else variables[0]
                default_values = get_default_variable_values(variables)
                min_variable_values = default_values.copy()
                min_variable_values[scan_variable] = scan_values[min_idx]
                print(f"\nVisualizing minimum energy configuration...")
                calculate_energy(min_variable_values, zmatrix_template, molecule_name, 
                               visualize=True, print_coords_flag=True)
        else:
            if args.recalculate and os.path.exists(pes_data_file):
                print(f"Recalculating PES data ('--recalculate' specified)...")
            elif not os.path.exists(pes_data_file):
                print(f"PES data file ('{pes_data_file}') not found. Calculating PES...")
            
            scan_range = tuple(args.scan_range) if args.scan_range else None
            scan_values, rel_energies = create_pes(
                zmatrix_file=args.zmatrix_file,
                scan_variable=args.scan_variable,
                scan_range=scan_range,
                num_points=args.num_points,
                molecule_name=molecule_name,
                visualize_configs=args.visualize_min_config,
                save_individual_plots=args.save_step_plots
            )
        
        print(f"\\nPES generation/loading complete. Data is in '{pes_data_file}'.")
        print("To perform TDA, run 2_ph.py using this data file (e.g., python src/2_ph.py {}).".format(pes_data_file))
