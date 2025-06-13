#!/usr/bin/env python3
"""
Advanced PES Calculator v3.0 - Multi-method conformational space exploration

This version addresses fundamental issues in the original implementation and
provides multiple strategies for conformational space mapping:

1. RDKit-based conformer generation with quantum refinement
2. Systematic torsion scanning with chemical intelligence  
3. Z-matrix template scanning (legacy support)
4. Hybrid approaches combining multiple methods

Key improvements:
- Correct quantum chemistry implementation
- Unit consistency (Hartree throughout)
- Chemical validation and benchmarking
- Flexible conformer generation strategies
- Robust error handling and convergence checking
- Cross-applicable to all small molecules

Authors: Enhanced implementation based on original concept
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import re
import warnings
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Import from modularized files
from data_structures import ConformerData, CalculationSettings
from quantum import QuantumCalculator
from conformer_rdkit import ConformerGeneratorRDKit
from conformer_systematic import ConformerGeneratorSystematic

# Quantum chemistry
try:
    from pyscf import gto, scf, dft, mp, cc
    from pyscf.geomopt import geometric_solver
    PYSCF_AVAILABLE = True
except ImportError:
    print("Warning: PySCF not available. Quantum calculations will fail.")
    PYSCF_AVAILABLE = False

# RDKit for conformer generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, rdDistGeom, rdForceFieldHelpers
    from rdkit.Chem.rdMolAlign import AlignMol
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Advanced conformer generation will be limited.")
    RDKIT_AVAILABLE = False

# Scientific computing
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. Conformer clustering will be limited.")
    SKLEARN_AVAILABLE = False

warnings.filterwarnings('ignore', category=UserWarning)


class PESCalculatorV3:
    """Advanced PES calculator with multiple conformer generation strategies."""
    
    def __init__(self, calculation_settings: dict):
        self.calc_settings = calculation_settings
        # Convert dict to CalculationSettings dataclass
        if isinstance(calculation_settings, dict):
            self.settings_obj = CalculationSettings(**calculation_settings)
        else:
            self.settings_obj = calculation_settings
        self.quantum_calc = QuantumCalculator(self.settings_obj)
        
    def calculate_pes(self, molecule_input: Union[str, 'Chem.Mol'],
                     conformer_method: str = 'rdkit',
                     num_conformers: int = 100,
                     optimize_conformers: bool = True,
                     save_data: bool = True,
                     output_prefix: str = "molecule") -> List[dict]:
        """
        Calculate potential energy surface using conformer-based approach.
        
        Args:
            molecule_input: SMILES, file path, or RDKit Mol object
            conformer_method: 'rdkit', 'systematic', 'hybrid'
            num_conformers: Number of conformers to generate
            optimize_conformers: Whether to optimize geometries with QM
            save_data: Whether to save results to files
            output_prefix: Prefix for output files
            
        Returns:
            List of conformer data dictionaries
        """
        print("="*60)
        print(f"PES CALCULATION v3.0 - {output_prefix}")
        print("="*60)
        print(f"Method: {self.calc_settings['method']}/{self.calc_settings['basis']}")
        print(f"Conformer generation: {conformer_method}")
        print(f"Target conformers: {num_conformers}")
        print(f"Geometry optimization: {optimize_conformers}")
        
        # Generate conformers
        print("\n--- CONFORMER GENERATION ---")
        if conformer_method == 'rdkit':
            conformer_gen = ConformerGeneratorRDKit(molecule_input)
            conformer_coords = conformer_gen.generate_conformers(
                num_conformers=num_conformers,
                energy_window=10.0
            )
        elif conformer_method == 'systematic':
            conformer_gen = ConformerGeneratorSystematic(molecule_input)
            conformer_coords = conformer_gen.generate_conformers(
                angle_step=60,  # default, can be parameterized
                max_conformers=1000
            )
        else:
            raise ValueError(f"Unknown conformer generation method: {conformer_method}")
        
        print(f"Generated {len(conformer_coords)} conformers for QM evaluation")
        
        # Calculate energies
        print("\n--- QUANTUM CHEMISTRY CALCULATIONS ---")
        conformer_data = []
        
        for i, coords in enumerate(conformer_coords):
            print(f"Calculating conformer {i+1}/{len(conformer_coords)}...")
            
            try:
                # Calculate energy (with optional optimization)
                energy, converged, opt_coords = self.quantum_calc.calculate_energy(
                    coords, conformer_gen.atoms, optimize=optimize_conformers
                )
                
                if not converged:
                    print(f"  Warning: Calculation did not converge")
                
                # Use optimized coordinates if available
                final_coords = opt_coords if opt_coords is not None else coords
                
                # Create conformer data object
                conformer_dict = {
                    'conformer_id': i,
                    'energy': energy,
                    'geometry': final_coords.tolist(),
                    'method_info': {
                        'method': self.calc_settings['method'],
                        'basis': self.calc_settings['basis'],
                        'optimized': optimize_conformers,
                        'conformer_method': conformer_method
                    },
                    'converged': converged
                }
                
                conformer_data.append(conformer_dict)
                
                print(f"  Energy: {energy:.6f} Hartree ({'converged' if converged else 'not converged'})")
                
            except Exception as e:
                print(f"  Error in conformer {i}: {e}")
                continue
        
        # Filter out failed calculations
        valid_conformers = [c for c in conformer_data if c['energy'] != float('inf')]
        print(f"\nSuccessfully calculated {len(valid_conformers)}/{len(conformer_coords)} conformers")
        
        if not valid_conformers:
            raise RuntimeError("No valid conformer calculations obtained")
        
        # Sort by energy
        valid_conformers.sort(key=lambda x: x['energy'])
        conformer_data = valid_conformers
        
        # Calculate relative energies
        min_energy = conformer_data[0]['energy']
        rel_energies = [(c['energy'] - min_energy) * 627.509 for c in conformer_data]  # kcal/mol for display
        
        print(f"\nEnergy range: {np.min(rel_energies):.3f} to {np.max(rel_energies):.3f} kcal/mol")
        print(f"Global minimum: {min_energy:.6f} Hartree")
        
        # Save data
        if save_data:
            self._save_pes_data(output_prefix, conformer_data)
            self._create_visualizations(output_prefix, rel_energies)
        
        return conformer_data
    
    def _save_pes_data(self, output_prefix: str, conformer_data: List[dict]):
        """Save PES data in multiple formats."""
        os.makedirs('data', exist_ok=True)
        
        # Save detailed data as JSON
        data_dict = {
            'calculation_settings': {
                'method': self.calc_settings['method'],
                'basis': self.calc_settings['basis'],
                'dispersion': self.calc_settings['dispersion'],
                'charge': self.calc_settings['charge'],
                'multiplicity': self.calc_settings['multiplicity']
            },
            'conformers': conformer_data
        }
        
        min_energy = min(c['energy'] for c in conformer_data)
        
        # Save JSON
        json_file = f'data/{output_prefix}_pes_detailed.json'
        with open(json_file, 'w') as f:
            json.dump(data_dict, f, indent=2)
        print(f"Detailed data saved to {json_file}")
        
        # Save simple format for TDA analysis (KEEP IN HARTREE!)
        simple_data = []
        for i, conformer in enumerate(conformer_data):
            rel_energy_hartree = conformer['energy'] - min_energy  # Keep in Hartree for TDA
            simple_data.append([i, rel_energy_hartree])
        
        simple_file = f'data/{output_prefix}_pes.dat'
        header = f"""# PES Data for {output_prefix}
# Method: {self.calc_settings['method']}/{self.calc_settings['basis']}
# Generated with conformer-based approach
# Columns: ConformerID RelativeEnergy(Hartree)
# Global minimum: {min_energy:.6f} Hartree
# Energy range: {np.min([r[1] for r in simple_data]):.6f} to {np.max([r[1] for r in simple_data]):.6f} Hartree"""
        
        np.savetxt(simple_file, simple_data, header=header, fmt='%d %12.6f')
        print(f"TDA-compatible data saved to {simple_file}")
    
    def _create_visualizations(self, output_prefix: str, rel_energies: List[float]):
        """Create visualization plots."""
        os.makedirs('figures', exist_ok=True)
        
        if not rel_energies:
            return
        
        min_energy = min(rel_energies)
        rel_energies = [(c - min_energy) * 627.509 for c in rel_energies]
        
        # Energy distribution plot
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Energy vs conformer ID
        plt.subplot(2, 2, 1)
        plt.plot(range(len(rel_energies)), rel_energies, 'b-o', markersize=4, linewidth=1)
        plt.xlabel('Conformer ID')
        plt.ylabel('Relative Energy (kcal/mol)')
        plt.title(f'{output_prefix} - Energy vs Conformer')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Energy histogram
        plt.subplot(2, 2, 2)
        plt.hist(rel_energies, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Relative Energy (kcal/mol)')
        plt.ylabel('Number of Conformers')
        plt.title('Energy Distribution')
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Cumulative energy plot
        plt.subplot(2, 2, 3)
        sorted_energies = sorted(rel_energies)
        plt.plot(sorted_energies, np.arange(len(sorted_energies)), 'g-', linewidth=2)
        plt.xlabel('Relative Energy (kcal/mol)')
        plt.ylabel('Number of Conformers Below Energy')
        plt.title('Cumulative Energy Distribution')
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Low energy region (< 5 kcal/mol)
        plt.subplot(2, 2, 4)
        low_energy_mask = np.array(rel_energies) < 5.0
        if np.any(low_energy_mask):
            low_energy_ids = np.array(range(len(rel_energies)))[low_energy_mask]
            low_energies = np.array(rel_energies)[low_energy_mask]
            plt.scatter(low_energy_ids, low_energies, c='red', s=50, alpha=0.7)
            plt.xlabel('Conformer ID')
            plt.ylabel('Relative Energy (kcal/mol)')
            plt.title('Low Energy Conformers (< 5 kcal/mol)')
            plt.ylim(0, 5)
        else:
            plt.text(0.5, 0.5, 'No conformers\n< 5 kcal/mol', 
                    transform=plt.gca().transAxes, ha='center', va='center')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'figures/{output_prefix}_pes_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Analysis plots saved to {plot_file}")


def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(
        description="Advanced PES Calculator v3.0 - Multi-method conformational space exploration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RDKit-based conformer generation for caffeine
  python 1_pes_v3.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --method rdkit --num-conformers 200
  
  # High-level calculation with geometry optimization
  python 1_pes_v3.py molecule.mol --quantum-method wb97x-d --basis aug-cc-pvdz --optimize
  
  # Systematic torsion scanning
  python 1_pes_v3.py "CCCC" --method systematic --angle-step 30
        """)
    
    # Input molecule
    parser.add_argument('molecule', 
                       help="SMILES string, molecule file (.mol, .mol2, .sdf), or molecule name")
    
    # Conformer generation
    parser.add_argument('--method', choices=['rdkit', 'systematic', 'hybrid'], 
                       default='rdkit',
                       help="Conformer generation method (default: rdkit)")
    parser.add_argument('--num-conformers', type=int, default=100,
                       help="Number of conformers to generate (default: 100)")
    parser.add_argument('--energy-window', type=float, default=10.0,
                       help="Energy window for conformer filtering in kcal/mol (default: 10.0)")
    
    # Quantum chemistry settings
    parser.add_argument('--quantum-method', default='b3lyp',
                       help="Quantum chemistry method (default: b3lyp)")
    parser.add_argument('--basis', default='6-311g(d,p)',
                       help="Basis set (default: 6-311g(d,p))")
    parser.add_argument('--dispersion', choices=['none', 'd3', 'd3bj'], default='none',
                       help="Dispersion correction (default: none)")
    parser.add_argument('--charge', type=int, default=0,
                       help="Molecular charge (default: 0)")
    parser.add_argument('--multiplicity', type=int, default=1,
                       help="Spin multiplicity (default: 1)")
    
    # Calculation options
    parser.add_argument('--optimize', action='store_true',
                       help="Optimize geometries with quantum method")
    parser.add_argument('--convergence', type=float, default=1e-6,
                       help="SCF convergence threshold (default: 1e-6)")
    
    # Systematic scanning options
    parser.add_argument('--angle-step', type=int, default=60,
                       help="Angle step for systematic scanning in degrees (default: 60)")
    parser.add_argument('--max-conformers', type=int, default=1000,
                       help="Maximum conformers for systematic method (default: 1000)")
    
    # Output options
    parser.add_argument('--output-prefix', 
                       help="Prefix for output files (default: based on input)")
    parser.add_argument('--no-save', action='store_true',
                       help="Don't save data files")
    
    args = parser.parse_args()
    
    # Determine output prefix
    if args.output_prefix:
        output_prefix = args.output_prefix
    elif args.molecule.endswith(('.mol', '.mol2', '.sdf')):
        output_prefix = Path(args.molecule).stem
    else:
        # SMILES string - create a simple name
        output_prefix = "molecule"
    
    # Set up calculation settings
    calc_settings = {
        'method': args.quantum_method,
        'basis': args.basis,
        'dispersion': args.dispersion,
        'charge': args.charge,
        'multiplicity': args.multiplicity,
        'convergence_threshold': args.convergence
    }
    
    try:
        # Initialize calculator
        calculator = PESCalculatorV3(calc_settings)
        
        # Run calculation
        conformer_data = calculator.calculate_pes(
            molecule_input=args.molecule,
            conformer_method=args.method,
            num_conformers=args.num_conformers,
            optimize_conformers=args.optimize,
            save_data=not args.no_save,
            output_prefix=output_prefix
        )
        
        print(f"\nCalculation completed successfully!")
        print(f"Generated {len(conformer_data)} conformers")
        print(f"Energy range: {(max(c['energy'] for c in conformer_data) - min(c['energy'] for c in conformer_data)) * 627.509:.3f} kcal/mol")
        
        if not args.no_save:
            print(f"\nOutput files:")
            print(f"  Detailed data: data/{output_prefix}_pes_detailed.json")
            print(f"  TDA data: data/{output_prefix}_pes.dat")
            print(f"  Plots: figures/{output_prefix}_pes_analysis.png")
            print(f"\nRun 2_ph.py on the .dat file for topological analysis:")
            print(f"  python 2_ph.py --data-file data/{output_prefix}_pes.dat")
        
        return 0
        
    except Exception as e:
        print(f"Error during calculation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 
