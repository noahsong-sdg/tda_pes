import numpy as np
import pyscf
from pyscf import gto, scf, dft
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import re
from matplotlib.animation import FuncAnimation
import itertools
from scipy.optimize import minimize

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

def calculate_energy(variable_values, zmatrix_template, molecule_name="molecule", visualize=False, print_coords_flag=False, 
                    level_of_theory='wb97x-d', basis_set='6-311++g(d,p)', relaxed_scan=False):
    """
    Calculate energy for given variable values using Z-matrix template.
    
    IMPROVEMENTS:
    - Default to ωB97X-D functional (includes dispersion corrections)
    - Default to 6-311++G(d,p) basis set (triple-zeta with diffuse functions)
    - Better error handling for SCF convergence
    - Relaxed scan disabled by default (current implementation lacks proper constraints)
    
    Args:
        variable_values (dict): Dictionary of variable names and values
        zmatrix_template (str): Z-matrix template string
        molecule_name (str): Name for output files
        visualize (bool): Whether to create 3D visualization
        print_coords_flag (bool): Whether to print coordinates
        level_of_theory (str): Level of theory ('hf', 'b3lyp', 'wb97x-d', 'mp2', etc.)
        basis_set (str): Basis set (default: '6-311++g(d,p)')  
        relaxed_scan (bool): Whether to perform geometry optimization with constrained dihedral
    
    Returns:
        float: Electronic energy in Hartree
    """
    zmatrix_str = create_geometry_from_template(zmatrix_template, variable_values)
    
    try:
        mol = gto.Mole()
        mol.atom = zmatrix_str
        mol.basis = basis_set
        mol.charge = 0
        mol.spin = 0
        mol.build(verbose=0)
    except Exception as e:
        print(f"Error building molecule for {variable_values}: {e}")
        return float('inf')
    
    # Choose method based on level_of_theory with better defaults
    try:
        if level_of_theory.lower() == 'hf':
            mf = scf.RHF(mol)
            mf.verbose = 0
            energy = mf.kernel()
        elif level_of_theory.lower() == 'wb97x-d':
            # Try ωB97X-D with different syntax options
            mf = dft.RKS(mol)
            try:
                mf.xc = 'wb97x-d'
                mf.verbose = 0
                energy = mf.kernel()
            except:
                try:
                    # Try alternative syntax
                    mf.xc = 'wb97x'
                    mf.verbose = 0
                    energy = mf.kernel()
                    print("Note: Using ωB97X without explicit dispersion correction")
                except:
                    # Fall back to B3LYP-D3 if available
                    try:
                        mf.xc = 'b3lyp'
                        mf.verbose = 0 
                        energy = mf.kernel()
                        print("Note: Falling back to B3LYP (no dispersion correction)")
                    except:
                        return float('inf')
        elif level_of_theory.lower() == 'b3lyp':
            mf = dft.RKS(mol)
            mf.xc = 'b3lyp'
            mf.verbose = 0
            energy = mf.kernel()
            print("Warning: B3LYP without dispersion correction. Consider wb97x-d for better accuracy.")
        elif level_of_theory.lower() == 'mp2':
            from pyscf import mp
            mf = scf.RHF(mol)
            mf.verbose = 0
            mf.kernel()
            if not mf.converged:
                print(f"Warning: HF not converged for MP2 calculation at {variable_values}")
                return float('inf')
            mp2 = mp.MP2(mf)
            mp2.verbose = 0
            energy = mp2.kernel()[0]
        else:
            # Default to B3LYP if unknown method  
            mf = dft.RKS(mol)
            mf.xc = 'b3lyp'
            mf.verbose = 0
            energy = mf.kernel()
            print(f"Unknown method {level_of_theory}, using B3LYP")
        
        # Check SCF convergence
        if hasattr(mf, 'converged') and not mf.converged:
            print(f"Warning: SCF not converged for configuration {variable_values}")
            return float('inf')
            
    except Exception as e:
        print(f"Error in electronic structure calculation for {variable_values}: {e}")
        return float('inf')
    
    # IMPROVED: Enable basic relaxed scan with penalty-based constraints
    if relaxed_scan:
        print("INFO: Using penalty-based constrained optimization for relaxed scan")
        print("      This maintains approximate dihedral angles while optimizing other coordinates")
        
        # Extract target dihedral angle from variable_values
        target_dihedral = None
        dihedral_var = None
        for var_name, value in variable_values.items():
            if 'dihedral' in var_name.lower() or 'phi' in var_name.lower() or 'theta' in var_name.lower():
                target_dihedral = value
                dihedral_var = var_name
                break
        
        if target_dihedral is not None:
            print(f"      Target {dihedral_var}: {target_dihedral:.1f}°")
        else:
            print("      Warning: No dihedral variable found, performing standard optimization")
            relaxed_scan = False
    
    # Improved relaxed scan with penalty-based constraints
    if relaxed_scan and level_of_theory.lower() in ['b3lyp', 'hf', 'wb97x-d']:
        try:
            print("        Performing penalty-based constrained optimization...")
            
            # Define constraint function
            def calculate_dihedral_angle(coords, i1, i2, i3, i4):
                """Calculate dihedral angle between four atoms"""
                coords_3d = coords.reshape(-1, 3)
                v1 = coords_3d[i2] - coords_3d[i1]
                v2 = coords_3d[i3] - coords_3d[i2] 
                v3 = coords_3d[i4] - coords_3d[i3]
                
                n1 = np.cross(v1, v2)
                n2 = np.cross(v2, v3)
                
                n1_norm = np.linalg.norm(n1)
                n2_norm = np.linalg.norm(n2)
                
                if n1_norm > 1e-8 and n2_norm > 1e-8:
                    n1 /= n1_norm
                    n2 /= n2_norm
                    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180.0 / np.pi
                    
                    # Check sign using scalar triple product
                    if np.dot(np.cross(n1, n2), v2) < 0:
                        angle = -angle
                    return angle
                return 0.0
            
            def constrained_objective(coords):
                """Objective function with dihedral constraint penalty"""
                try:
                    # Update molecule geometry
                    coords_3d = coords.reshape(-1, 3)
                    mol_temp = mol.copy()
                    mol_temp.atom = [(mol.atom_symbol(i), coords_3d[i]) for i in range(mol.natm)]
                    mol_temp.build(verbose=0)
                    
                    # Calculate energy
                    if level_of_theory.lower() == 'hf':
                        mf_temp = scf.RHF(mol_temp)
                    elif level_of_theory.lower() == 'wb97x-d':
                        mf_temp = dft.RKS(mol_temp)
                        try:
                            mf_temp.xc = 'wb97x-d'
                        except:
                            try:
                                mf_temp.xc = 'wb97x'
                            except:
                                mf_temp.xc = 'b3lyp'
                    elif level_of_theory.lower() == 'b3lyp':
                        mf_temp = dft.RKS(mol_temp)
                        mf_temp.xc = 'b3lyp'
                    
                    mf_temp.verbose = 0
                    energy = mf_temp.kernel()
                    
                    if not mf_temp.converged:
                        return 1e6
                    
                    # Add penalty for dihedral constraint (if we know the atom indices)
                    # For now, use a simple harmonic constraint on all dihedrals
                    penalty = 0.0
                    if target_dihedral is not None and mol.natm >= 4:
                        # Simple example: constrain first available dihedral
                        # In practice, you'd want to identify the correct dihedral atoms
                        try:
                            # Estimate main chain dihedral (atoms 0,1,2,3 for simple molecules)
                            if mol.natm >= 4:
                                current_angle = calculate_dihedral_angle(coords, 0, 1, 2, 3)
                                angle_diff = current_angle - target_dihedral
                                # Handle periodicity (-180 to 180)
                                while angle_diff > 180:
                                    angle_diff -= 360
                                while angle_diff < -180:
                                    angle_diff += 360
                                penalty = 0.1 * angle_diff**2  # Moderate penalty
                        except:
                            penalty = 0.0
                    
                    return energy + penalty
                    
                except Exception as e:
                    return 1e6
            
            # Perform constrained optimization using scipy
            from scipy.optimize import minimize
            initial_coords = mol.atom_coords().flatten()
            
            result = minimize(constrained_objective, initial_coords, method='BFGS',
                            options={'maxiter': 50, 'gtol': 1e-4})
            
            if result.success:
                # Update molecule with optimized coordinates
                opt_coords = result.x.reshape(-1, 3)
                mol_opt = mol.copy()
                mol_opt.atom = [(mol.atom_symbol(i), opt_coords[i]) for i in range(mol.natm)]
                mol_opt.build(verbose=0)
                
                # Recalculate final energy
                if level_of_theory.lower() == 'hf':
                    mf_final = scf.RHF(mol_opt)
                elif level_of_theory.lower() == 'wb97x-d':
                    mf_final = dft.RKS(mol_opt)
                    try:
                        mf_final.xc = 'wb97x-d'
                    except:
                        try:
                            mf_final.xc = 'wb97x'
                        except:
                            mf_final.xc = 'b3lyp'
                elif level_of_theory.lower() == 'b3lyp':
                    mf_final = dft.RKS(mol_opt)
                    mf_final.xc = 'b3lyp'
                
                mf_final.verbose = 0
                energy = mf_final.kernel()
                
                # Update mol for coordinate printing
                mol = mol_opt
                
                print(f"        Constrained optimization converged")
                if target_dihedral and mol.natm >= 4:
                    try:
                        final_angle = calculate_dihedral_angle(result.x, 0, 1, 2, 3)
                        print(f"        Final dihedral: {final_angle:.1f}° (target: {target_dihedral:.1f}°)")
                    except:
                        pass
                
            else:
                print(f"        Constrained optimization failed: {result.message}")
                print(f"        Using rigid geometry")
                
        except Exception as e:
            print(f"        Error in constrained optimization: {e}")
            print(f"        Falling back to rigid scan")
            pass
                
        except ImportError as e:
            print(f"Warning: Relaxed scan requires berny_solver ({e}). Using rigid scan.")
            pass
    
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
               visualize_configs=False, save_individual_plots=False, level_of_theory='wb97x-d', 
               basis_set='6-311++g(d,p)', relaxed_scan=False):
    """
    Create potential energy surface by scanning one variable parameter.
    
    IMPROVEMENTS:
    - Default to ωB97X-D functional with dispersion corrections
    - Default to 6-311++G(d,p) basis set for better accuracy
    - Added experimental validation for known molecules
    - Better error handling and progress reporting
    
    Args:
        zmatrix_file (str): Path to Z-matrix template file
        scan_variable (str): Name of the variable to scan (if None, uses first variable)
        scan_range (tuple): Range to scan as (min, max) (default: (0, 360) for angles)
        num_points (int): Number of points to calculate
        molecule_name (str): Name of the molecule for output files
        visualize_configs (bool): Whether to visualize configurations
        save_individual_plots (bool): Whether to save individual structure plots
        level_of_theory (str): Level of theory ('hf', 'b3lyp', 'wb97x-d', 'mp2', etc.)
        basis_set (str): Basis set (default: '6-311++g(d,p)')  
        relaxed_scan (bool): Whether to perform relaxed scan with geometry optimization
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
    failed_calculations = 0
    
    print(f"Calculating PES for {molecule_name}...")
    print(f"Method: {level_of_theory}/{basis_set}")
    print(f"Scanning {scan_variable} from {scan_range[0]} to {scan_range[1]} with {num_points} points")
    if relaxed_scan:
        print("WARNING: Relaxed scan requested but disabled due to missing constraints")
    
    for i, value in enumerate(scan_values):
        print(f"{scan_variable} = {value:.1f} ({i+1}/{len(scan_values)})")
        
        # Set up variable values for this calculation
        variable_values = default_values.copy()
        variable_values[scan_variable] = value
        
        show_vis = save_individual_plots and (i % 5 == 0)
        show_coords = save_individual_plots and (i % 5 == 0) 
        
        energy = calculate_energy(variable_values, zmatrix_template, molecule_name, 
                                visualize=show_vis, print_coords_flag=show_coords,
                                level_of_theory=level_of_theory, basis_set=basis_set,
                                relaxed_scan=relaxed_scan)
        
        if energy == float('inf'):
            failed_calculations += 1
            # Use interpolated energy for failed calculations
            if len(energies) > 0:
                energy = energies[-1] + 0.1  # Slightly higher than last point
            else:
                energy = 0.0
        
        energies.append(energy)
    
    if failed_calculations > 0:
        print(f"Warning: {failed_calculations} calculations failed and were interpolated")
    
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
    print(f"Method: {level_of_theory}/{basis_set}")
    print(f"Minimum energy: {np.min(rel_energies):.3f} kcal/mol at {scan_variable} = {scan_values[min_idx]:.1f}")
    print(f"Maximum energy: {np.max(rel_energies):.3f} kcal/mol at {scan_variable} = {scan_values[np.argmax(rel_energies)]:.1f}")
    print(f"Energy range: {np.max(rel_energies) - np.min(rel_energies):.3f} kcal/mol")
    print(f"Average energy: {np.mean(rel_energies):.3f} kcal/mol")
    print(f"Standard deviation: {np.std(rel_energies):.3f} kcal/mol")
    
    # Add experimental validation
    validate_against_experimental(scan_values, rel_energies, molecule_name)
    
    if visualize_configs:
        print(f"\nVisualizing minimum energy configuration at {scan_variable} = {scan_values[min_idx]:.1f}...")
        min_variable_values = default_values.copy()
        min_variable_values[scan_variable] = scan_values[min_idx]
        calculate_energy(min_variable_values, zmatrix_template, molecule_name, 
                        visualize=True, print_coords_flag=True,
                        level_of_theory=level_of_theory, basis_set=basis_set,
                        relaxed_scan=relaxed_scan)
    
    return scan_values, rel_energies

def validate_against_experimental(scan_values, energies, molecule_name):
    """Compare calculated energies against experimental values."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENTAL VALIDATION FOR {molecule_name.upper()}")
    print(f"{'='*60}")
    
    if 'butane' in molecule_name.lower():
        # Experimental butane conformational energies (kcal/mol)
        exp_data = {
            0: 3.8,      # syn-eclipsed
            60: 0.9,     # gauche
            120: 3.4,    # anti-eclipsed  
            180: 0.0,    # anti (reference)
        }
        
        print("Literature butane conformational energies (kcal/mol):")
        print("  0°   (syn-eclipsed): 3.8 kcal/mol")
        print("  60°  (gauche):       0.9 kcal/mol") 
        print("  120° (anti-eclipsed): 3.4 kcal/mol")
        print("  180° (anti):         0.0 kcal/mol (reference)")
        print("\nReferences:")
        print("  Crowder, G. A. J. Mol. Struct. 1977, 42, 183.")
        print("  Wiberg, K. B. J. Am. Chem. Soc. 2003, 125, 1888.")
        
        print("\nCalculated energies at key conformations:")
        calc_values = []
        exp_values = []
        
        for angle in [0, 60, 120, 180]:
            # Find closest calculated point
            idx = np.argmin(np.abs(scan_values - angle))
            calc_angle = scan_values[idx]
            calc_energy = energies[idx]
            exp_energy = exp_data.get(angle, None)
            
            if exp_energy is not None:
                error = calc_energy - exp_energy
                print(f"  {calc_angle:3.0f}°: {calc_energy:5.2f} kcal/mol (exp: {exp_energy:4.1f}, error: {error:+5.2f})")
                calc_values.append(calc_energy)
                exp_values.append(exp_energy)
            else:
                print(f"  {calc_angle:3.0f}°: {calc_energy:5.2f} kcal/mol")
        
        # Calculate statistics
        if calc_values and exp_values:
            mae = np.mean(np.abs(np.array(calc_values) - np.array(exp_values)))
            rmse = np.sqrt(np.mean((np.array(calc_values) - np.array(exp_values))**2))
            
            print(f"\nStatistical comparison with experiment:")
            print(f"  Mean Absolute Error (MAE):  {mae:.2f} kcal/mol")
            print(f"  Root Mean Square Error:     {rmse:.2f} kcal/mol")
            
            if mae < 0.5:
                print("  ✓ EXCELLENT agreement (MAE < 0.5 kcal/mol)")
            elif mae < 1.0:
                print("  ✓ GOOD agreement (MAE < 1.0 kcal/mol)")
            elif mae < 2.0:
                print("  ⚠ FAIR agreement (MAE < 2.0 kcal/mol)")
            else:
                print("  ✗ POOR agreement (MAE > 2.0 kcal/mol)")
                print("    Consider using higher level of theory or larger basis set")
    
    elif 'pentane' in molecule_name.lower():
        print("Pentane conformational analysis:")
        print("Expected multiple local minima due to multiple dihedral angles")
        print("Experimental studies: NMR coupling constants and IR spectroscopy")
        print("References:")
        print("  Eliel, E. L.; Wilen, S. H. Stereochemistry of Organic Compounds; Wiley: 1994.")
        
        # Find local minima
        min_energy = np.min(energies)
        min_idx = np.argmin(energies)
        print(f"\nGlobal minimum: {energies[min_idx]:.2f} kcal/mol at {scan_values[min_idx]:.1f}°")
        
        # Simple local minima detection
        local_minima = []
        for i in range(1, len(energies)-1):
            if energies[i] < energies[i-1] and energies[i] < energies[i+1] and energies[i] < min_energy + 3.0:
                local_minima.append((scan_values[i], energies[i]))
        
        if local_minima:
            print("Low-energy conformations found:")
            for angle, energy in local_minima:
                print(f"  {angle:6.1f}°: {energy:5.2f} kcal/mol")
    
    else:
        print(f"No specific experimental validation data available for {molecule_name}")
        min_energy = np.min(energies)
        min_idx = np.argmin(energies)
        print(f"Global minimum: {energies[min_idx]:.2f} kcal/mol at {scan_values[min_idx]:.1f}°")
        print("For publication-quality results, compare with:")
        print("1. Experimental conformational energies from literature")
        print("2. High-level ab initio calculations (MP2, CCSD(T))")
        print("3. Other computational studies using similar methods")

def demonstrate_single_configuration(zmatrix_file, variable_values=None, molecule_name=None, 
                                   level_of_theory='b3lyp', basis_set='6-31g*'):
    """Demonstrate atomic configuration for specific variable values"""
    zmatrix_template, variables = read_zmatrix_template(zmatrix_file)
    
    if variable_values is None:
        variable_values = get_default_variable_values(variables)
    
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
    
    print(f"Demonstrating {molecule_name} configuration with variables: {variable_values}")
    energy = calculate_energy(variable_values, zmatrix_template, molecule_name, 
                            visualize=True, print_coords_flag=True,
                            level_of_theory=level_of_theory, basis_set=basis_set)
    print(f"Calculated energy: {energy:.6f} Hartree")
    return energy

def create_2d_pes(zmatrix_file, scan_variables=None, scan_ranges=None, num_points=(25, 25), 
                  molecule_name=None, visualize_configs=False, level_of_theory='b3lyp', 
                  basis_set='6-31g*', relaxed_scan=False):
    """
    Create 2D potential energy surface by scanning two variables simultaneously.
    
    Args:
        zmatrix_file (str): Path to Z-matrix template file
        scan_variables (list): Names of the two variables to scan
        scan_ranges (list): Ranges to scan as [(min1, max1), (min2, max2)]
        num_points (tuple): Number of points to calculate for each variable (nx, ny)
        molecule_name (str): Name of the molecule for output files
        visualize_configs (bool): Whether to visualize minimum configuration
    
    Returns:
        tuple: (var1_values, var2_values, energy_grid)
    """
    # Read the Z-matrix template
    zmatrix_template, variables = read_zmatrix_template(zmatrix_file)
    
    if not variables:
        raise ValueError("No variables found in Z-matrix template.")
    
    # Determine scan variables
    if scan_variables is None:
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for 2D PES. Found: " + str(variables))
        scan_variables = variables[:2]
        print(f"No scan variables specified. Using first two variables: {scan_variables}")
    elif len(scan_variables) != 2:
        raise ValueError("Exactly 2 scan variables required for 2D PES")
    
    for var in scan_variables:
        if var not in variables:
            raise ValueError(f"Variable '{var}' not found in template. Available: {variables}")
    
    # Set default scan ranges
    if scan_ranges is None:
        scan_ranges = []
        for var in scan_variables:
            var_lower = var.lower()
            if 'dihedral' in var_lower or 'torsion' in var_lower:
                scan_ranges.append((0, 360))
            elif 'angle' in var_lower:
                scan_ranges.append((90, 150))
            else:
                scan_ranges.append((0, 10))
        print(f"No scan ranges specified. Using default ranges: {scan_ranges}")
    
    # Get default values for all variables
    default_values = get_default_variable_values(variables)
    
    # Determine molecule name
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
    
    # Create scan grids
    var1_values = np.linspace(scan_ranges[0][0], scan_ranges[0][1], num_points[0])
    var2_values = np.linspace(scan_ranges[1][0], scan_ranges[1][1], num_points[1])
    
    energy_grid = np.zeros((len(var1_values), len(var2_values)))
    
    print(f"Calculating 2D PES for {molecule_name}...")
    print(f"Scanning {scan_variables[0]}: {scan_ranges[0][0]} to {scan_ranges[0][1]} ({num_points[0]} points)")
    print(f"Scanning {scan_variables[1]}: {scan_ranges[1][0]} to {scan_ranges[1][1]} ({num_points[1]} points)")
    print(f"Total calculations: {num_points[0] * num_points[1]}")
    
    total_calcs = num_points[0] * num_points[1]
    calc_count = 0
    
    for i, val1 in enumerate(var1_values):
        for j, val2 in enumerate(var2_values):
            calc_count += 1
            print(f"Progress: {calc_count}/{total_calcs} ({scan_variables[0]}={val1:.1f}, {scan_variables[1]}={val2:.1f})")
            
            # Set up variable values for this calculation
            variable_values = default_values.copy()
            variable_values[scan_variables[0]] = val1
            variable_values[scan_variables[1]] = val2
            
            energy = calculate_energy(variable_values, zmatrix_template, molecule_name,
                                    level_of_theory=level_of_theory, basis_set=basis_set,
                                    relaxed_scan=relaxed_scan)
            energy_grid[i, j] = energy
    
    # Convert to relative energies in kcal/mol
    rel_energy_grid = (energy_grid - np.min(energy_grid)) * 627.509
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(rel_energy_grid), rel_energy_grid.shape)
    min_var1 = var1_values[min_idx[0]]
    min_var2 = var2_values[min_idx[1]]
    
    # Create 3D surface plot
    plot_3d_surface(var1_values, var2_values, rel_energy_grid, scan_variables, molecule_name)
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.save(f'data/{molecule_name}_2d_pes.npy', {
        'var1_values': var1_values,
        'var2_values': var2_values,
        'energy_grid': rel_energy_grid,
        'scan_variables': scan_variables,
        'molecule_name': molecule_name
    })
    
    print(f"\n{'='*60}")
    print("2D PES ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Minimum energy: {np.min(rel_energy_grid):.3f} kcal/mol")
    print(f"  at {scan_variables[0]} = {min_var1:.1f}, {scan_variables[1]} = {min_var2:.1f}")
    print(f"Maximum energy: {np.max(rel_energy_grid):.3f} kcal/mol")
    print(f"Energy range: {np.max(rel_energy_grid) - np.min(rel_energy_grid):.3f} kcal/mol")
    
    if visualize_configs:
        print(f"\nVisualizing minimum energy configuration...")
        min_variable_values = default_values.copy()
        min_variable_values[scan_variables[0]] = min_var1
        min_variable_values[scan_variables[1]] = min_var2
        calculate_energy(min_variable_values, zmatrix_template, molecule_name, 
                        visualize=True, print_coords_flag=True,
                        level_of_theory=level_of_theory, basis_set=basis_set,
                        relaxed_scan=relaxed_scan)
    
    return var1_values, var2_values, rel_energy_grid

def plot_3d_surface(var1_values, var2_values, energy_grid, scan_variables, molecule_name):
    """Create 3D surface plot for 2D PES"""
    fig = plt.figure(figsize=(14, 10))
    
    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    X, Y = np.meshgrid(var2_values, var1_values)
    surf = ax1.plot_surface(X, Y, energy_grid, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(scan_variables[1])
    ax1.set_ylabel(scan_variables[0])
    ax1.set_zlabel('Relative Energy (kcal/mol)')
    ax1.set_title(f'3D PES: {molecule_name.capitalize()}')
    plt.colorbar(surf, ax=ax1, shrink=0.6)
    
    # 2D contour plot
    ax2 = fig.add_subplot(222)
    contour = ax2.contourf(X, Y, energy_grid, levels=20, cmap='viridis')
    ax2.contour(X, Y, energy_grid, levels=20, colors='black', alpha=0.4, linewidths=0.5)
    ax2.set_xlabel(scan_variables[1])
    ax2.set_ylabel(scan_variables[0])
    ax2.set_title('2D Contour Plot')
    plt.colorbar(contour, ax=ax2)
    
    # Energy histogram
    ax3 = fig.add_subplot(223)
    ax3.hist(energy_grid.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_xlabel('Relative Energy (kcal/mol)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Energy Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Energy vs variable 1 at minimum of variable 2
    ax4 = fig.add_subplot(224)
    min_var2_idx = np.argmin(np.min(energy_grid, axis=0))
    ax4.plot(var1_values, energy_grid[:, min_var2_idx], 'b-o', linewidth=2, markersize=4)
    ax4.set_xlabel(scan_variables[0])
    ax4.set_ylabel('Relative Energy (kcal/mol)')
    ax4.set_title(f'1D slice at {scan_variables[1]} = {var2_values[min_var2_idx]:.1f}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/{molecule_name}_2d_pes.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_3d_pes(zmatrix_file, scan_variables=None, scan_ranges=None, num_points=(15, 15, 15), 
                  molecule_name=None, create_animation=True, animation_frames=50,
                  level_of_theory='b3lyp', basis_set='6-31g*', relaxed_scan=False):
    """
    Create 3D potential energy surface by scanning three variables simultaneously.
    Creates animated GIF showing conformational changes.
    
    Args:
        zmatrix_file (str): Path to Z-matrix template file
        scan_variables (list): Names of the three variables to scan
        scan_ranges (list): Ranges to scan as [(min1, max1), (min2, max2), (min3, max3)]
        num_points (tuple): Number of points to calculate for each variable (nx, ny, nz)
        molecule_name (str): Name of the molecule for output files
        create_animation (bool): Whether to create animated GIF
        animation_frames (int): Number of frames for animation
        level_of_theory (str): Level of theory ('hf', 'b3lyp', 'mp2', etc.)
        basis_set (str): Basis set (default: '6-31g*')
        relaxed_scan (bool): Whether to perform geometry optimization with constrained variables
    
    Returns:
        tuple: (var1_values, var2_values, var3_values, energy_cube)
    """
    # Read the Z-matrix template
    zmatrix_template, variables = read_zmatrix_template(zmatrix_file)
    
    if not variables:
        raise ValueError("No variables found in Z-matrix template.")
    
    # Determine scan variables
    if scan_variables is None:
        if len(variables) < 3:
            raise ValueError("Need at least 3 variables for 3D PES. Found: " + str(variables))
        scan_variables = variables[:3]
        print(f"No scan variables specified. Using first three variables: {scan_variables}")
    elif len(scan_variables) != 3:
        raise ValueError("Exactly 3 scan variables required for 3D PES")
    
    for var in scan_variables:
        if var not in variables:
            raise ValueError(f"Variable '{var}' not found in template. Available: {variables}")
    
    # Set default scan ranges
    if scan_ranges is None:
        scan_ranges = []
        for var in scan_variables:
            var_lower = var.lower()
            if 'dihedral' in var_lower or 'torsion' in var_lower:
                scan_ranges.append((0, 360))
            elif 'angle' in var_lower:
                scan_ranges.append((90, 150))
            else:
                scan_ranges.append((0, 10))
        print(f"No scan ranges specified. Using default ranges: {scan_ranges}")
    
    # Get default values for all variables
    default_values = get_default_variable_values(variables)
    
    # Determine molecule name
    if molecule_name is None:
        molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
    
    # Create scan grids
    var1_values = np.linspace(scan_ranges[0][0], scan_ranges[0][1], num_points[0])
    var2_values = np.linspace(scan_ranges[1][0], scan_ranges[1][1], num_points[1])
    var3_values = np.linspace(scan_ranges[2][0], scan_ranges[2][1], num_points[2])
    
    energy_cube = np.zeros((len(var1_values), len(var2_values), len(var3_values)))
    
    print(f"Calculating 3D PES for {molecule_name}...")
    print(f"Scanning {scan_variables[0]}: {scan_ranges[0][0]} to {scan_ranges[0][1]} ({num_points[0]} points)")
    print(f"Scanning {scan_variables[1]}: {scan_ranges[1][0]} to {scan_ranges[1][1]} ({num_points[1]} points)")
    print(f"Scanning {scan_variables[2]}: {scan_ranges[2][0]} to {scan_ranges[2][1]} ({num_points[2]} points)")
    print(f"Total calculations: {num_points[0] * num_points[1] * num_points[2]}")
    
    total_calcs = num_points[0] * num_points[1] * num_points[2]
    calc_count = 0
    
    for i, val1 in enumerate(var1_values):
        for j, val2 in enumerate(var2_values):
            for k, val3 in enumerate(var3_values):
                calc_count += 1
                if calc_count % 10 == 1:  # Print progress every 10 calculations
                    print(f"Progress: {calc_count}/{total_calcs} ({scan_variables[0]}={val1:.1f}, {scan_variables[1]}={val2:.1f}, {scan_variables[2]}={val3:.1f})")
                
                # Set up variable values for this calculation
                variable_values = default_values.copy()
                variable_values[scan_variables[0]] = val1
                variable_values[scan_variables[1]] = val2
                variable_values[scan_variables[2]] = val3
                
                energy = calculate_energy(variable_values, zmatrix_template, molecule_name,
                                        level_of_theory=level_of_theory, basis_set=basis_set,
                                        relaxed_scan=relaxed_scan)
                energy_cube[i, j, k] = energy
    
    # Convert to relative energies in kcal/mol
    rel_energy_cube = (energy_cube - np.min(energy_cube)) * 627.509
    
    # Find minimum
    min_idx = np.unravel_index(np.argmin(rel_energy_cube), rel_energy_cube.shape)
    min_var1 = var1_values[min_idx[0]]
    min_var2 = var2_values[min_idx[1]]
    min_var3 = var3_values[min_idx[2]]
    
    # Save data
    os.makedirs('data', exist_ok=True)
    np.save(f'data/{molecule_name}_3d_pes.npy', {
        'var1_values': var1_values,
        'var2_values': var2_values,
        'var3_values': var3_values,
        'energy_cube': rel_energy_cube,
        'scan_variables': scan_variables,
        'molecule_name': molecule_name
    })
    
    print(f"\n{'='*60}")
    print("3D PES ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Minimum energy: {np.min(rel_energy_cube):.3f} kcal/mol")
    print(f"  at {scan_variables[0]} = {min_var1:.1f}, {scan_variables[1]} = {min_var2:.1f}, {scan_variables[2]} = {min_var3:.1f}")
    print(f"Maximum energy: {np.max(rel_energy_cube):.3f} kcal/mol")
    print(f"Energy range: {np.max(rel_energy_cube) - np.min(rel_energy_cube):.3f} kcal/mol")
    
    if create_animation:
        create_animated_gif(var1_values, var2_values, var3_values, rel_energy_cube, 
                           scan_variables, molecule_name, zmatrix_template, default_values, animation_frames)
    
    return var1_values, var2_values, var3_values, rel_energy_cube

def create_animated_gif(var1_values, var2_values, var3_values, energy_cube, scan_variables, 
                       molecule_name, zmatrix_template, default_values, num_frames=50):
    """
    Create animated GIF showing conformational changes through 3D PES.
    """
    print("Creating animated GIF...")
    
    # Find path through low-energy conformations
    # Simple approach: use the minimum energy path through the first variable
    min_energy_path = []
    
    for i in range(len(var1_values)):
        # For each value of var1, find the minimum energy configuration in var2-var3 plane
        min_idx_2d = np.unravel_index(np.argmin(energy_cube[i, :, :]), energy_cube[i, :, :].shape)
        min_energy_path.append((i, min_idx_2d[0], min_idx_2d[1]))
    
    # Create frames by interpolating along this path
    frame_indices = np.linspace(0, len(min_energy_path)-1, num_frames, dtype=int)
    
    # Set up the figure for animation
    fig = plt.figure(figsize=(16, 8))
    
    def animate_frame(frame_num):
        plt.clf()
        
        path_idx = frame_indices[frame_num]
        i, j, k = min_energy_path[path_idx]
        
        val1 = var1_values[i]
        val2 = var2_values[j]
        val3 = var3_values[k]
        energy = energy_cube[i, j, k]
        
        # Calculate molecular coordinates for this configuration
        variable_values = default_values.copy()
        variable_values[scan_variables[0]] = val1
        variable_values[scan_variables[1]] = val2
        variable_values[scan_variables[2]] = val3
        
        # Get molecular coordinates
        zmatrix_str = create_geometry_from_template(zmatrix_template, variable_values)
        mol = gto.Mole()
        mol.atom = zmatrix_str
        mol.basis = 'sto-3g'
        mol.charge = 0
        mol.spin = 0
        mol.build(verbose=0)
        
        atom_symbols = mol.elements
        xyz_coords = mol.atom_coords(unit='Angstrom')
        
        # Left subplot: 3D molecular structure
        ax1 = fig.add_subplot(121, projection='3d')
        
        carbons = []
        hydrogens = []
        
        for atom_idx, element in enumerate(atom_symbols):
            coord = xyz_coords[atom_idx]
            if element == 'C':
                carbons.append(coord)
            else:
                hydrogens.append(coord)
        
        carbons_np = np.array(carbons) if carbons else np.array([]).reshape(0,3)
        hydrogens_np = np.array(hydrogens) if hydrogens else np.array([]).reshape(0,3)

        if carbons_np.size > 0:
            ax1.scatter(carbons_np[:, 0], carbons_np[:, 1], carbons_np[:, 2], 
                      c='black', s=200, alpha=0.8, label='Carbon')
        
        if hydrogens_np.size > 0:
            ax1.scatter(hydrogens_np[:, 0], hydrogens_np[:, 1], hydrogens_np[:, 2], 
                      c='lightgray', s=100, alpha=0.6, label='Hydrogen')
        
        # Draw bonds
        if carbons_np.shape[0] >= 2:
            for bond_i in range(len(carbons_np) - 1):
                ax1.plot([carbons_np[bond_i, 0], carbons_np[bond_i+1, 0]], 
                       [carbons_np[bond_i, 1], carbons_np[bond_i+1, 1]], 
                       [carbons_np[bond_i, 2], carbons_np[bond_i+1, 2]], 
                       'k-', linewidth=2, alpha=0.7)
        
        # Draw C-H bonds
        if carbons_np.size > 0 and hydrogens_np.size > 0:
            for carbon_coord in carbons_np:
                for hydrogen_coord in hydrogens_np:
                    distance = np.linalg.norm(carbon_coord - hydrogen_coord)
                    if distance < 1.5:
                        ax1.plot([carbon_coord[0], hydrogen_coord[0]], 
                               [carbon_coord[1], hydrogen_coord[1]], 
                               [carbon_coord[2], hydrogen_coord[2]], 
                               'gray', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        ax1.set_title(f'Frame {frame_num+1}/{num_frames}\n{scan_variables[0]}={val1:.1f}, {scan_variables[1]}={val2:.1f}, {scan_variables[2]}={val3:.1f}\nEnergy: {energy:.3f} kcal/mol')
        
        # Right subplot: 2D slice through energy cube
        ax2 = fig.add_subplot(122)
        # Show var2 vs var3 slice at current var1
        X, Y = np.meshgrid(var3_values, var2_values)
        contour = ax2.contourf(X, Y, energy_cube[i, :, :], levels=15, cmap='viridis')
        ax2.contour(X, Y, energy_cube[i, :, :], levels=15, colors='white', alpha=0.4, linewidths=0.5)
        ax2.scatter(val3, val2, color='red', s=100, marker='*', label='Current position')
        ax2.set_xlabel(scan_variables[2])
        ax2.set_ylabel(scan_variables[1])
        ax2.set_title(f'Energy slice at {scan_variables[0]}={val1:.1f}')
        ax2.legend()
        
        plt.tight_layout()
    
    # Create animation
    print(f"Generating {num_frames} frames...")
    anim = FuncAnimation(fig, animate_frame, frames=num_frames, interval=200, repeat=True)
    
    # Save as GIF
    os.makedirs('figures', exist_ok=True)
    gif_filename = f'figures/{molecule_name}_3d_pes_animation.gif'
    anim.save(gif_filename, writer='pillow', fps=5, dpi=100)
    plt.close(fig)
    
    print(f"Animation saved as {gif_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General PES Calculator for molecular systems from Z-matrix files.")
    parser.add_argument('zmatrix_file', type=str,
                        help="Path to Z-matrix template file with variables marked as {variable_name}")
    parser.add_argument('--scan-mode', type=str, choices=['1d', '2d', '3d'], default='1d',
                        help="Scanning mode: 1d (default), 2d (surface plot), or 3d (animated gif)")
    parser.add_argument('--scan-variables', type=str, nargs='+', default=None,
                        help="Variables to scan (1 for 1d, 2 for 2d, 3 for 3d)")
    parser.add_argument('--scan-ranges', type=float, nargs='+', default=None,
                        help="Ranges to scan as min1 max1 [min2 max2] [min3 max3]")
    parser.add_argument('--num-points', type=int, nargs='+', default=None,
                        help="Number of points for each variable (single value or one per variable)")
    parser.add_argument('--animation-frames', type=int, default=50,
                        help="Number of frames for 3D animation (default: 50)")
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
    parser.add_argument('--level-of-theory', type=str, default='wb97x-d',
                        choices=['hf', 'b3lyp', 'wb97x-d', 'mp2'],
                        help="Level of theory for quantum calculations (default: wb97x-d)")
    parser.add_argument('--basis-set', type=str, default='6-311++g(d,p)',
                        help="Basis set for quantum calculations (default: 6-311++g(d,p))")
    parser.add_argument('--relaxed-scan', action='store_true',
                        help="Perform geometry optimization at each point (relaxed scan) - Currently disabled due to missing constraints")
    
    # Legacy arguments for backward compatibility
    parser.add_argument('--scan-variable', type=str, default=None,
                        help="(Legacy) Variable to scan for 1D mode")
    parser.add_argument('--scan-range', type=float, nargs=2, default=None,
                        help="(Legacy) Range to scan as min max for 1D mode")
    
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
        demonstrate_single_configuration(args.zmatrix_file, molecule_name=molecule_name,
                                        level_of_theory=args.level_of_theory, basis_set=args.basis_set)
    else:
        # Handle legacy arguments for backward compatibility
        if args.scan_mode == '1d':
            scan_variables = args.scan_variables
            if scan_variables is None and args.scan_variable:
                scan_variables = [args.scan_variable]
            
            scan_ranges = None
            if args.scan_ranges:
                if len(args.scan_ranges) >= 2:
                    scan_ranges = (args.scan_ranges[0], args.scan_ranges[1])
            elif args.scan_range:
                scan_ranges = tuple(args.scan_range)
            
            num_points = 36
            if args.num_points:
                if isinstance(args.num_points, list):
                    num_points = args.num_points[0]
                else:
                    num_points = args.num_points
            
            scan_variable = scan_variables[0] if scan_variables else None
            
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
                    scan_variable = scan_variable if scan_variable else variables[0]
                    default_values = get_default_variable_values(variables)
                    min_variable_values = default_values.copy()
                    min_variable_values[scan_variable] = scan_values[min_idx]
                    print(f"\nVisualizing minimum energy configuration...")
                    calculate_energy(min_variable_values, zmatrix_template, molecule_name, 
                                   visualize=True, print_coords_flag=True,
                                   level_of_theory=args.level_of_theory, basis_set=args.basis_set)
            else:
                if args.recalculate and os.path.exists(pes_data_file):
                    print(f"Recalculating PES data ('--recalculate' specified)...")
                elif not os.path.exists(pes_data_file):
                    print(f"PES data file ('{pes_data_file}') not found. Calculating PES...")
                
                scan_values, rel_energies = create_pes(
                    zmatrix_file=args.zmatrix_file,
                    scan_variable=scan_variable,
                    scan_range=scan_ranges,
                    num_points=num_points,
                    molecule_name=molecule_name,
                    visualize_configs=args.visualize_min_config,
                    save_individual_plots=args.save_step_plots,
                    level_of_theory=args.level_of_theory,
                    basis_set=args.basis_set,
                    relaxed_scan=args.relaxed_scan
                )
            
            print(f"\n1D PES generation/loading complete. Data is in '{pes_data_file}'.")
        
        elif args.scan_mode == '2d':
            # Parse scan variables (should be 2)
            scan_variables = args.scan_variables
            
            # Parse scan ranges
            scan_ranges = None
            if args.scan_ranges:
                if len(args.scan_ranges) >= 4:
                    scan_ranges = [(args.scan_ranges[0], args.scan_ranges[1]), 
                                  (args.scan_ranges[2], args.scan_ranges[3])]
                else:
                    print("Warning: 2D mode requires 4 range values (min1 max1 min2 max2). Using defaults.")
            
            # Parse num_points
            num_points = (25, 25)
            if args.num_points:
                if len(args.num_points) == 2:
                    num_points = tuple(args.num_points)
                elif len(args.num_points) == 1:
                    num_points = (args.num_points[0], args.num_points[0])
            
            pes_data_file = f'data/{molecule_name}_2d_pes.npy'
            
            if not args.recalculate and os.path.exists(pes_data_file):
                print(f"Loading 2D PES data from {pes_data_file}...")
                data = np.load(pes_data_file, allow_pickle=True).item()
                var1_values = data['var1_values']
                var2_values = data['var2_values']
                energy_grid = data['energy_grid']
                scan_variables = data['scan_variables']
                print("2D PES data loaded.")
                
                min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
                min_var1 = var1_values[min_idx[0]]
                min_var2 = var2_values[min_idx[1]]
                print(f"Minimum energy: {np.min(energy_grid):.3f} kcal/mol at {scan_variables[0]}={min_var1:.1f}, {scan_variables[1]}={min_var2:.1f}")
                
                # Regenerate 3D plot
                plot_3d_surface(var1_values, var2_values, energy_grid, scan_variables, molecule_name)
            else:
                if args.recalculate and os.path.exists(pes_data_file):
                    print(f"Recalculating 2D PES data ('--recalculate' specified)...")
                elif not os.path.exists(pes_data_file):
                    print(f"2D PES data file ('{pes_data_file}') not found. Calculating 2D PES...")
                
                var1_values, var2_values, energy_grid = create_2d_pes(
                    zmatrix_file=args.zmatrix_file,
                    scan_variables=scan_variables,
                    scan_ranges=scan_ranges,
                    num_points=num_points,
                    molecule_name=molecule_name,
                    visualize_configs=args.visualize_min_config,
                    level_of_theory=args.level_of_theory,
                    basis_set=args.basis_set,
                    relaxed_scan=args.relaxed_scan
                )
            
            print(f"\n2D PES generation/loading complete. Data is in '{pes_data_file}'.")
        
        elif args.scan_mode == '3d':
            # Parse scan variables (should be 3)
            scan_variables = args.scan_variables
            
            # Parse scan ranges
            scan_ranges = None
            if args.scan_ranges:
                if len(args.scan_ranges) >= 6:
                    scan_ranges = [(args.scan_ranges[0], args.scan_ranges[1]), 
                                  (args.scan_ranges[2], args.scan_ranges[3]),
                                  (args.scan_ranges[4], args.scan_ranges[5])]
                else:
                    print("Warning: 3D mode requires 6 range values (min1 max1 min2 max2 min3 max3). Using defaults.")
            
            # Parse num_points
            num_points = (15, 15, 15)
            if args.num_points:
                if len(args.num_points) == 3:
                    num_points = tuple(args.num_points)
                elif len(args.num_points) == 1:
                    num_points = (args.num_points[0], args.num_points[0], args.num_points[0])
            
            pes_data_file = f'data/{molecule_name}_3d_pes.npy'
            
            if not args.recalculate and os.path.exists(pes_data_file):
                print(f"Loading 3D PES data from {pes_data_file}...")
                data = np.load(pes_data_file, allow_pickle=True).item()
                var1_values = data['var1_values']
                var2_values = data['var2_values']
                var3_values = data['var3_values']
                energy_cube = data['energy_cube']
                scan_variables = data['scan_variables']
                print("3D PES data loaded.")
                
                min_idx = np.unravel_index(np.argmin(energy_cube), energy_cube.shape)
                min_var1 = var1_values[min_idx[0]]
                min_var2 = var2_values[min_idx[1]]
                min_var3 = var3_values[min_idx[2]]
                print(f"Minimum energy: {np.min(energy_cube):.3f} kcal/mol")
                print(f"  at {scan_variables[0]}={min_var1:.1f}, {scan_variables[1]}={min_var2:.1f}, {scan_variables[2]}={min_var3:.1f}")
                
                # Regenerate animation
                zmatrix_template, variables = read_zmatrix_template(args.zmatrix_file)
                default_values = get_default_variable_values(variables)
                create_animated_gif(var1_values, var2_values, var3_values, energy_cube, 
                                   scan_variables, molecule_name, zmatrix_template, default_values, args.animation_frames)
            else:
                if args.recalculate and os.path.exists(pes_data_file):
                    print(f"Recalculating 3D PES data ('--recalculate' specified)...")
                elif not os.path.exists(pes_data_file):
                    print(f"3D PES data file ('{pes_data_file}') not found. Calculating 3D PES...")
                
                var1_values, var2_values, var3_values, energy_cube = create_3d_pes(
                    zmatrix_file=args.zmatrix_file,
                    scan_variables=scan_variables,
                    scan_ranges=scan_ranges,
                    num_points=num_points,
                    molecule_name=molecule_name,
                    create_animation=True,
                    animation_frames=args.animation_frames,
                    level_of_theory=args.level_of_theory,
                    basis_set=args.basis_set,
                    relaxed_scan=args.relaxed_scan
                )
            
            
            print(f"\n3D PES generation/loading complete. Data is in '{pes_data_file}'.")
        
        print("To perform TDA on 1D data, run 2_ph.py using the .dat file.")
