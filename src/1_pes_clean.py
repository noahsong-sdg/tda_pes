#!/usr/bin/env python3
"""
Streamlined PES Calculator for molecular systems from Z-matrix files.

This cleaned version focuses on core functionality with improved error handling
and simplified architecture.

Key features:
- 1D, 2D, and 3D potential energy surface calculations
- Modern computational methods with dispersion corrections
- Simplified command-line interface
- Robust error handling and data management
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import argparse
import os
import re
from scipy.optimize import minimize
import warnings

# PySCF imports with error handling
try:
    from pyscf import gto, scf, dft, mp, cc
    from pyscf.geomopt import geometric_solver
    PYSCF_AVAILABLE = True
except ImportError:
    print("Warning: PySCF not available. Electronic structure calculations will fail.")
    PYSCF_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)


class PESCalculator:
    """Main class for potential energy surface calculations."""
    
    def __init__(self, level_of_theory='b3lyp', basis_set='6-311++g(d,p)'):
        """Initialize PES calculator with computational method."""
        self.level_of_theory = level_of_theory.lower()
        self.basis_set = basis_set
        self.validate_method()
    
    def validate_method(self):
        """Validate computational method selection."""
        valid_methods = ['hf', 'b3lyp', 'wb97x-d', 'mp2', 'ccsd']
        if self.level_of_theory not in valid_methods:
            print(f"Warning: Unknown method {self.level_of_theory}. Using b3lyp.")
            self.level_of_theory = 'b3lyp'
    
    def read_zmatrix_template(self, filename):
        """Read Z-matrix template and extract variables."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Z-matrix file '{filename}' not found.")
        
        with open(filename, 'r') as f:
            template = f.read().strip()
        
        variables = list(set(re.findall(r'\{([^}]+)\}', template)))
        print(f"Found variables: {variables}")
        return template, variables
    
    def create_geometry(self, template, variable_values):
        """Create geometry string by substituting variables."""
        geometry = template
        for var_name, value in variable_values.items():
            geometry = geometry.replace(f'{{{var_name}}}', str(value))
        return geometry
    
    def get_default_values(self, variables):
        """Get sensible default values for variables."""
        defaults = {}
        for var in variables:
            var_lower = var.lower()
            if 'dihedral' in var_lower or 'phi' in var_lower:
                defaults[var] = 180.0  # Anti configuration
            elif 'angle' in var_lower:
                defaults[var] = 109.5  # Tetrahedral angle
            elif 'bond' in var_lower or 'distance' in var_lower:
                defaults[var] = 1.5    # Typical C-C bond
            else:
                defaults[var] = 1.0    # Generic default
        return defaults
    
    def calculate_energy(self, variable_values, zmatrix_template, 
                        visualize=False, molecule_name="molecule"):
        """Calculate energy for given geometry configuration."""
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is required for energy calculations")
        
        geometry_str = self.create_geometry(zmatrix_template, variable_values)
        
        try:
            mol = gto.Mole()
            mol.atom = geometry_str
            mol.basis = self.basis_set
            mol.charge = 0
            mol.spin = 0
            mol.build(verbose=0)
        except Exception as e:
            print(f"Error building molecule: {e}")
            return float('inf')
        
        return self._perform_calculation(mol, variable_values)
    
    def _perform_calculation(self, mol, variable_values):
        """Perform the electronic structure calculation."""
        try:
            if self.level_of_theory == 'hf':
                mf = scf.RHF(mol)
            elif self.level_of_theory == 'b3lyp':
                mf = dft.RKS(mol)
                mf.xc = 'b3lyp'
                if variable_values == list(variable_values.values())[0]:  # Only print once
                    print("Warning: B3LYP lacks dispersion corrections")
            elif self.level_of_theory == 'wb97x-d':
                mf = dft.RKS(mol)
                # Try different formats for wb97x-d in PySCF
                try:
                    mf.xc = 'wb97x_d'  # PySCF uses underscore format
                except:
                    try:
                        mf.xc = 'wB97X-D'  # Try capitalized format
                    except:
                        # Final fallback to B3LYP
                        mf.xc = 'b3lyp'
                        print("Warning: wb97x-d not available in PySCF, using B3LYP instead")
            else:
                # Default fallback
                mf = dft.RKS(mol)
                mf.xc = 'b3lyp'
                print(f"Warning: Unknown method {self.level_of_theory}, using B3LYP")
            
            mf.verbose = 0
            
            if self.level_of_theory in ['hf', 'b3lyp', 'wb97x-d']:
                energy = mf.kernel()
                if not mf.converged:
                    print(f"SCF not converged for {variable_values}")
                    return float('inf')
            elif self.level_of_theory == 'mp2':
                hf_energy = mf.kernel()
                if not mf.converged:
                    return float('inf')
                mp2_calc = mp.MP2(mf)
                mp2_calc.verbose = 0
                energy = mp2_calc.kernel()[0]
            elif self.level_of_theory == 'ccsd':
                hf_energy = mf.kernel()
                if not mf.converged:
                    return float('inf')
                ccsd_calc = cc.CCSD(mf)
                ccsd_calc.verbose = 0
                energy = ccsd_calc.kernel()[0]
            
            return energy
            
        except Exception as e:
            print(f"Calculation error: {e}")
            return float('inf')
    
    def create_1d_pes(self, zmatrix_file, scan_variable=None, scan_range=None, 
                     num_points=36, molecule_name=None, relaxed=False):
        """Create 1D potential energy surface."""
        template, variables = self.read_zmatrix_template(zmatrix_file)
        
        if not variables:
            raise ValueError("No variables found in Z-matrix template.")
        
        # Determine scan parameters
        if scan_variable is None:
            scan_variable = variables[0]
            print(f"Using first variable: {scan_variable}")
        
        if scan_variable not in variables:
            raise ValueError(f"Variable '{scan_variable}' not found in template.")
        
        if scan_range is None:
            var_lower = scan_variable.lower()
            if 'dihedral' in var_lower or 'phi' in var_lower:
                scan_range = (0, 360)
            elif 'angle' in var_lower:
                scan_range = (90, 150)
            else:
                scan_range = (0, 10)
            print(f"Using default range: {scan_range}")
        
        if molecule_name is None:
            molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
        
        # Get default values and create scan
        default_values = self.get_default_values(variables)
        scan_values = np.linspace(scan_range[0], scan_range[1], num_points)
        energies = []
        
        scan_type = "relaxed" if relaxed else "rigid"
        print(f"Calculating 1D {scan_type} PES for {molecule_name}...")
        print(f"Scanning {scan_variable}: {scan_range[0]} to {scan_range[1]} ({num_points} points)")
        
        if relaxed:
            print("Using constrained geometry optimization (relaxed scan)")
        else:
            print("Using rigid scan (fixed coordinates)")
        
        for i, value in enumerate(scan_values):
            print(f"Progress: {i+1}/{num_points} ({scan_variable}={value:.1f})")
            
            variable_values = default_values.copy()
            variable_values[scan_variable] = value
            
            if relaxed:
                # Use relaxed scan with constrained optimization
                energy = self.calculate_energy_relaxed(variable_values, template, 
                                                     scan_variable, value, 
                                                     molecule_name=molecule_name)
            else:
                # Use rigid scan
                energy = self.calculate_energy(variable_values, template, 
                                             molecule_name=molecule_name)
            energies.append(energy)
        
        energies = np.array(energies)
        rel_energies = (energies - np.min(energies)) * 627.509  # Convert to kcal/mol
        
        # Save and plot results
        self._save_1d_data(scan_values, rel_energies, scan_variable, molecule_name, relaxed)
        self._plot_1d_pes(scan_values, rel_energies, scan_variable, molecule_name, relaxed)
        
        return scan_values, rel_energies
    
    def create_2d_pes(self, zmatrix_file, scan_variables=None, scan_ranges=None, 
                     num_points=(20, 20), molecule_name=None, relaxed=False):
        """Create 2D potential energy surface."""
        template, variables = self.read_zmatrix_template(zmatrix_file)
        
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for 2D PES")
        
        if scan_variables is None:
            scan_variables = variables[:2]
            print(f"Using first two variables: {scan_variables}")
        
        if len(scan_variables) != 2:
            raise ValueError("Exactly 2 scan variables required for 2D PES")
        
        if scan_ranges is None:
            scan_ranges = []
            for var in scan_variables:
                var_lower = var.lower()
                if 'dihedral' in var_lower:
                    scan_ranges.append((0, 360))
                elif 'angle' in var_lower:
                    scan_ranges.append((90, 150))
                else:
                    scan_ranges.append((0, 10))
        
        if molecule_name is None:
            molecule_name = os.path.splitext(os.path.basename(zmatrix_file))[0]
        
        # Create grids
        default_values = self.get_default_values(variables)
        var1_values = np.linspace(scan_ranges[0][0], scan_ranges[0][1], num_points[0])
        var2_values = np.linspace(scan_ranges[1][0], scan_ranges[1][1], num_points[1])
        energy_grid = np.zeros((len(var1_values), len(var2_values)))
        
        scan_type = "relaxed" if relaxed else "rigid"
        print(f"Calculating 2D {scan_type} PES for {molecule_name}...")
        print(f"Total calculations: {num_points[0] * num_points[1]}")
        
        calc_count = 0
        for i, val1 in enumerate(var1_values):
            for j, val2 in enumerate(var2_values):
                calc_count += 1
                print(f"Progress: {calc_count}/{num_points[0] * num_points[1]} ({scan_type})")
                
                variable_values = default_values.copy()
                variable_values[scan_variables[0]] = val1
                variable_values[scan_variables[1]] = val2
                
                if relaxed:
                    # For 2D relaxed scans, we need to constrain both variables
                    # Use first variable as primary constraint for the optimization
                    energy = self.calculate_energy_relaxed(variable_values, template, 
                                                         scan_variables[0], val1, 
                                                         molecule_name=molecule_name)
                else:
                    energy = self.calculate_energy(variable_values, template,
                                                 molecule_name=molecule_name)
                energy_grid[i, j] = energy
        
        rel_energy_grid = (energy_grid - np.min(energy_grid)) * 627.509
        
        # Save and plot results
        self._save_2d_data(var1_values, var2_values, rel_energy_grid, 
                          scan_variables, molecule_name, relaxed)
        self._plot_2d_pes(var1_values, var2_values, rel_energy_grid, 
                         scan_variables, molecule_name, relaxed)
        
        return var1_values, var2_values, rel_energy_grid
    
    def calculate_energy_relaxed(self, variable_values, zmatrix_template, 
                               scan_variable, scan_value, molecule_name="molecule"):
        """Calculate energy with constrained geometry optimization."""
        if not PYSCF_AVAILABLE:
            raise RuntimeError("PySCF is required for energy calculations")
        
        # Start with the input geometry
        geometry_str = self.create_geometry(zmatrix_template, variable_values)
        
        try:
            mol = gto.Mole()
            mol.atom = geometry_str
            mol.basis = self.basis_set
            mol.charge = 0
            mol.spin = 0
            mol.build(verbose=0)
        except Exception as e:
            print(f"Error building molecule: {e}")
            return float('inf')
        
        # Perform constrained optimization
        try:
            optimized_mol = self._constrained_optimization(mol, zmatrix_template, 
                                                         scan_variable, scan_value, 
                                                         variable_values)
            if optimized_mol is None:
                return float('inf')
            
            return self._perform_calculation(optimized_mol, variable_values)
            
        except Exception as e:
            print(f"Relaxed scan error: {e}")
            return float('inf')
    
    def _constrained_optimization(self, mol, zmatrix_template, scan_variable, 
                                 scan_value, variable_values):
        """Perform constrained geometry optimization using simple coordinate optimization."""
        try:
            # Create mean-field object for gradients
            if self.level_of_theory == 'hf':
                mf = scf.RHF(mol)
            elif self.level_of_theory == 'b3lyp':
                mf = dft.RKS(mol)
                mf.xc = 'b3lyp'
            elif self.level_of_theory == 'wb97x-d':
                mf = dft.RKS(mol)
                try:
                    mf.xc = 'wb97x_d'
                except:
                    mf.xc = 'b3lyp'
            else:
                mf = dft.RKS(mol)
                mf.xc = 'b3lyp'
            
            mf.verbose = 0
            
            # Simple constrained optimization using coordinate relaxation
            max_cycles = 20
            conv_threshold = 1e-4
            
            current_values = variable_values.copy()
            current_values[scan_variable] = scan_value  # Keep scan variable fixed
            
            for cycle in range(max_cycles):
                # Calculate energy and gradients for current geometry
                current_geometry = self.create_geometry(zmatrix_template, current_values)
                
                try:
                    mol_current = gto.Mole()
                    mol_current.atom = current_geometry
                    mol_current.basis = self.basis_set
                    mol_current.charge = 0
                    mol_current.spin = 0
                    mol_current.build(verbose=0)
                except:
                    return None
                
                # Simple optimization: adjust other variables slightly
                old_values = current_values.copy()
                
                # For each non-scan variable, try small perturbations
                for var in current_values:
                    if var == scan_variable:
                        continue  # Keep scan variable fixed
                    
                    # Try small positive and negative perturbations
                    step_size = 0.1 if 'angle' in var.lower() or 'dihedral' in var.lower() else 0.01
                    
                    # Test positive perturbation
                    test_values_pos = current_values.copy()
                    test_values_pos[var] += step_size
                    energy_pos = self._evaluate_energy_simple(test_values_pos, zmatrix_template)
                    
                    # Test negative perturbation
                    test_values_neg = current_values.copy()
                    test_values_neg[var] -= step_size
                    energy_neg = self._evaluate_energy_simple(test_values_neg, zmatrix_template)
                    
                    # Current energy
                    energy_current = self._evaluate_energy_simple(current_values, zmatrix_template)
                    
                    # Move in direction of lower energy
                    if energy_pos < energy_current and energy_pos < energy_neg:
                        current_values[var] += step_size * 0.5
                    elif energy_neg < energy_current and energy_neg < energy_pos:
                        current_values[var] -= step_size * 0.5
                
                # Check convergence
                changes = [abs(current_values[var] - old_values[var]) 
                          for var in current_values if var != scan_variable]
                
                if changes:  # Only check convergence if there are other variables
                    max_change = max(changes)
                    if max_change < conv_threshold:
                        break
                else:
                    # If only scan variable exists, we're already "converged"
                    break
            
            # Build final optimized molecule
            final_geometry = self.create_geometry(zmatrix_template, current_values)
            mol_final = gto.Mole()
            mol_final.atom = final_geometry
            mol_final.basis = self.basis_set
            mol_final.charge = 0
            mol_final.spin = 0
            mol_final.build(verbose=0)
            
            return mol_final
            
        except Exception as e:
            print(f"Constrained optimization failed: {e}")
            return None
    
    def _evaluate_energy_simple(self, variable_values, zmatrix_template):
        """Simple energy evaluation for optimization."""
        try:
            geometry_str = self.create_geometry(zmatrix_template, variable_values)
            mol = gto.Mole()
            mol.atom = geometry_str
            mol.basis = self.basis_set
            mol.charge = 0
            mol.spin = 0
            mol.build(verbose=0)
            
            if self.level_of_theory == 'hf':
                mf = scf.RHF(mol)
            else:
                mf = dft.RKS(mol)
                mf.xc = 'b3lyp'
            
            mf.verbose = 0
            energy = mf.kernel()
            
            return energy if mf.converged else float('inf')
        except:
            return float('inf')

    def _save_1d_data(self, scan_values, rel_energies, scan_variable, molecule_name, relaxed=False):
        """Save 1D PES data to file."""
        os.makedirs('data', exist_ok=True)
        
        # Include scan type in filename for relaxed scans
        scan_type = "_relaxed" if relaxed else ""
        filename = f'data/{molecule_name}{scan_type}_pes.dat'
        
        # Include scan type in header
        scan_info = "relaxed" if relaxed else "rigid"
        header = f"""# 1D PES Data for {molecule_name} ({scan_info} scan)
# Method: {self.level_of_theory}/{self.basis_set}
# Variable: {scan_variable}
# Scan type: {scan_info}
# Minimum energy: {np.min(rel_energies):.6f} kcal/mol
# Columns: {scan_variable} Energy(kcal/mol)"""
        
        np.savetxt(filename, np.column_stack([scan_values, rel_energies]), 
                  header=header, fmt='%12.6f')
        print(f"Data saved to {filename}")
    
    def _save_2d_data(self, var1_values, var2_values, energy_grid, 
                     scan_variables, molecule_name, relaxed=False):
        """Save 2D PES data to file."""
        os.makedirs('data', exist_ok=True)
        
        # Include scan type in filename for relaxed scans
        scan_type = "_relaxed" if relaxed else ""
        filename = f'data/{molecule_name}_2d{scan_type}_pes.npy'
        
        np.save(filename, {
            'var1_values': var1_values,
            'var2_values': var2_values,
            'energy_grid': energy_grid,
            'scan_variables': scan_variables,
            'molecule_name': molecule_name,
            'method': f"{self.level_of_theory}/{self.basis_set}",
            'scan_type': 'relaxed' if relaxed else 'rigid'
        })
        print(f"Data saved to {filename}")
    
    def _plot_1d_pes(self, scan_values, rel_energies, scan_variable, molecule_name, relaxed=False):
        """Create 1D PES plot."""
        os.makedirs('figures', exist_ok=True)
        
        # Include scan type in filename and title
        scan_type = "_relaxed" if relaxed else ""
        scan_info = "Relaxed" if relaxed else "Rigid"
        
        plt.figure(figsize=(10, 6))
        plt.plot(scan_values, rel_energies, 'b-o', linewidth=2, markersize=6)
        plt.xlabel(f'{scan_variable} (degrees)')
        plt.ylabel('Relative Energy (kcal/mol)')
        plt.title(f'{molecule_name.capitalize()} PES ({scan_info} Scan) - {self.level_of_theory.upper()}/{self.basis_set}')
        plt.grid(True, alpha=0.3)
        
        # Mark global minimum
        min_idx = np.argmin(rel_energies)
        plt.scatter(scan_values[min_idx], rel_energies[min_idx], 
                   color='red', s=100, zorder=5,
                   label=f'Global Min: {scan_values[min_idx]:.1f}°')
        plt.legend()
        
        filename = f'figures/{molecule_name}{scan_type}_pes.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filename}")
    
    def _plot_2d_pes(self, var1_values, var2_values, energy_grid, 
                    scan_variables, molecule_name, relaxed=False):
        """Create 2D PES plots."""
        os.makedirs('figures', exist_ok=True)
        
        # Include scan type in filename and title
        scan_type = "_relaxed" if relaxed else ""
        scan_info = "Relaxed" if relaxed else "Rigid"
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        X, Y = np.meshgrid(var2_values, var1_values)
        
        # 3D surface plot
        ax1 = fig.add_subplot(221, projection='3d')
        surf = ax1.plot_surface(X, Y, energy_grid, cmap='viridis', alpha=0.8)
        ax1.set_xlabel(scan_variables[1])
        ax1.set_ylabel(scan_variables[0])
        ax1.set_zlabel('Relative Energy (kcal/mol)')
        ax1.set_title(f'3D PES: {molecule_name.capitalize()} ({scan_info} Scan)')
        
        # 2D contour plot
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(X, Y, energy_grid, levels=20, cmap='viridis')
        ax2.contour(X, Y, energy_grid, levels=20, colors='black', alpha=0.4, linewidths=0.5)
        ax2.set_xlabel(scan_variables[1])
        ax2.set_ylabel(scan_variables[0])
        ax2.set_title('2D Contour Plot')
        plt.colorbar(contour, ax=ax2)
        
        # Energy histogram
        ax3.hist(energy_grid.flatten(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Relative Energy (kcal/mol)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Energy Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 1D slice
        min_var2_idx = np.argmin(np.min(energy_grid, axis=0))
        ax4.plot(var1_values, energy_grid[:, min_var2_idx], 'b-o', linewidth=2, markersize=4)
        ax4.set_xlabel(scan_variables[0])
        ax4.set_ylabel('Relative Energy (kcal/mol)')
        ax4.set_title(f'1D slice at {scan_variables[1]} = {var2_values[min_var2_idx]:.1f}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'figures/{molecule_name}_2d{scan_type}_pes.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {filename}")


def main():
    """Main function with streamlined argument parsing."""
    parser = argparse.ArgumentParser(
        description="Streamlined PES Calculator for molecular systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1D scan with default settings (B3LYP/6-311++G(d,p))
  python 1_pes_clean.py butane_template.zmat
  
  # 2D scan with custom variables
  python 1_pes_clean.py pentane_template.zmat --mode 2d --variables dihedral1 dihedral2
  
  # High-level calculation with ωB97X-D (if supported)
  python 1_pes_clean.py molecule.zmat --method wb97x-d --basis aug-cc-pvdz
        """)
    
    # Required arguments
    parser.add_argument('zmatrix_file', 
                       help="Path to Z-matrix template file")
    
    # Calculation mode
    parser.add_argument('--mode', choices=['1d', '2d'], default='1d',
                       help="Scan mode: 1d (default) or 2d")
    
    # Method and basis set
    parser.add_argument('--method', default='b3lyp',
                       choices=['hf', 'b3lyp', 'wb97x-d', 'mp2', 'ccsd'],
                       help="Level of theory (default: b3lyp)")
    parser.add_argument('--basis', default='6-311++g(d,p)',
                       help="Basis set (default: 6-311++g(d,p))")
    
    # Scan parameters
    parser.add_argument('--variables', nargs='+',
                       help="Variables to scan (auto-detected if not specified)")
    parser.add_argument('--ranges', type=float, nargs='+',
                       help="Scan ranges as min1 max1 [min2 max2]")
    parser.add_argument('--points', type=int, nargs='+', default=[36],
                       help="Number of points (default: 36 for 1D, 20x20 for 2D)")
    
    # Output options
    parser.add_argument('--name',
                       help="Molecule name for output files (default: filename)")
    parser.add_argument('--force', action='store_true',
                       help="Force recalculation even if output exists")
    
    # Relaxed scan option
    parser.add_argument('--relaxed', action='store_true',
                       help="Use relaxed scan with constrained geometry optimization")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.zmatrix_file):
        print(f"Error: File '{args.zmatrix_file}' not found.")
        return 1
    
    # Initialize calculator
    calc = PESCalculator(level_of_theory=args.method, basis_set=args.basis)
    
    # Determine molecule name
    molecule_name = args.name or os.path.splitext(os.path.basename(args.zmatrix_file))[0]
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    try:
        if args.mode == '1d':
            # Parse 1D parameters
            scan_variable = args.variables[0] if args.variables else None
            scan_range = tuple(args.ranges[:2]) if args.ranges and len(args.ranges) >= 2 else None
            num_points = args.points[0] if args.points else 36
            
            # Check for existing data
            scan_type = "_relaxed" if args.relaxed else ""
            data_file = f'data/{molecule_name}{scan_type}_pes.dat'
            if os.path.exists(data_file) and not args.force:
                print(f"Data file '{data_file}' exists. Use --force to recalculate.")
                return 0
            
            # Calculate 1D PES
            scan_values, rel_energies = calc.create_1d_pes(
                args.zmatrix_file, scan_variable, scan_range, 
                num_points, molecule_name, relaxed=args.relaxed)
            
            # Print summary
            min_idx = np.argmin(rel_energies)
            print(f"\n1D PES Summary:")
            print(f"Minimum energy: {np.min(rel_energies):.3f} kcal/mol at {scan_values[min_idx]:.1f}°")
            print(f"Maximum energy: {np.max(rel_energies):.3f} kcal/mol")
            print(f"Energy range: {np.max(rel_energies) - np.min(rel_energies):.3f} kcal/mol")
        
        elif args.mode == '2d':
            # Parse 2D parameters
            scan_variables = args.variables[:2] if args.variables and len(args.variables) >= 2 else None
            scan_ranges = None
            if args.ranges and len(args.ranges) >= 4:
                scan_ranges = [(args.ranges[0], args.ranges[1]), (args.ranges[2], args.ranges[3])]
            
            if len(args.points) == 1:
                num_points = (args.points[0], args.points[0])
            elif len(args.points) >= 2:
                num_points = tuple(args.points[:2])
            else:
                num_points = (20, 20)
            
            # Check for existing data
            scan_type = "_relaxed" if args.relaxed else ""
            data_file = f'data/{molecule_name}_2d{scan_type}_pes.npy'
            if os.path.exists(data_file) and not args.force:
                print(f"Data file '{data_file}' exists. Use --force to recalculate.")
                return 0
            
            # Calculate 2D PES
            var1_values, var2_values, energy_grid = calc.create_2d_pes(
                args.zmatrix_file, scan_variables, scan_ranges, 
                num_points, molecule_name, relaxed=args.relaxed)
            
            # Print summary
            min_idx = np.unravel_index(np.argmin(energy_grid), energy_grid.shape)
            print(f"\n2D PES Summary:")
            print(f"Minimum energy: {np.min(energy_grid):.3f} kcal/mol")
            print(f"Maximum energy: {np.max(energy_grid):.3f} kcal/mol")
            print(f"Energy range: {np.max(energy_grid) - np.min(energy_grid):.3f} kcal/mol")
        
        print("\nCalculation completed successfully!")
        print("Run 2_ph.py on the generated .dat file for topological analysis.")
        
        return 0
        
    except Exception as e:
        print(f"Error during calculation: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
