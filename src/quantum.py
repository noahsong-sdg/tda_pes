import numpy as np
from data_structures import CalculationSettings

# Quantum chemistry imports
try:
    from pyscf import gto, scf, dft, mp, cc
    from pyscf.geomopt import geometric_solver
    PYSCF_AVAILABLE = True
except ImportError:
    print("Warning: PySCF not available. Quantum calculations will fail.")
    PYSCF_AVAILABLE = False

class QuantumCalculator:
    """
    Handles quantum chemistry calculations with proper error handling.
    """
    def __init__(self, settings: CalculationSettings):
        self.settings = settings
        self._validate_settings()
    
    def _validate_settings(self):
        if not PYSCF_AVAILABLE:
            raise ImportError("PySCF is required for quantum calculations")
        valid_methods = ['hf', 'b3lyp', 'pbe', 'pbe0', 'wb97x-d', 'mp2', 'ccsd']
        if self.settings.method.lower() not in valid_methods:
            raise ValueError(f"Method {self.settings.method} not supported. "
                             f"Valid methods: {valid_methods}")
    
    def calculate_energy(self, coordinates: np.ndarray, atoms: list, optimize: bool = False):
        """
        Calculate energy for given geometry.
        Returns:
            energy (float): Energy in Hartree
            converged (bool): Whether calculation converged
            opt_coords (np.ndarray): Optimized coordinates if optimize=True
        """
        try:
            mol = self._build_molecule(coordinates, atoms)
            calc = self._setup_calculation(mol)
            if optimize:
                try:
                    mol_eq = geometric_solver.optimize(calc)
                    energy = calc.e_tot
                    opt_coords = mol_eq.atom_coords()
                    converged = calc.converged
                except Exception as e:
                    print(f"Optimization failed: {e}")
                    energy = calc.kernel()
                    opt_coords = coordinates
                    converged = calc.converged
            else:
                energy = calc.kernel()
                opt_coords = None
                converged = calc.converged
            return energy, converged, opt_coords
        except Exception as e:
            print(f"Quantum calculation failed: {e}")
            return float('inf'), False, None
    
    def _build_molecule(self, coordinates: np.ndarray, atoms: list):
        atom_string = []
        for atom, coord in zip(atoms, coordinates):
            atom_string.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        mol = gto.Mole()
        mol.atom = '; '.join(atom_string)
        mol.basis = self.settings.basis
        mol.charge = self.settings.charge
        mol.spin = self.settings.multiplicity - 1
        mol.build(verbose=0)
        return mol
    
    def _setup_calculation(self, mol):
        method = self.settings.method.lower()
        if method == 'hf':
            calc = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        elif method in ['mp2']:
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            mf.conv_tol = self.settings.convergence_threshold
            mf.max_cycle = self.settings.max_iterations
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("SCF did not converge for MP2 calculation")
            calc = mp.MP2(mf)
            return calc
        elif method in ['ccsd']:
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
            mf.conv_tol = self.settings.convergence_threshold
            mf.max_cycle = self.settings.max_iterations
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("SCF did not converge for CCSD calculation")
            calc = cc.CCSD(mf)
            return calc
        else:
            calc = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            calc.xc = method
            if self.settings.dispersion == 'd3':
                try:
                    calc = calc.apply(dft.D3Disp(mol))
                except:
                    print("Warning: D3 dispersion correction not available")
            elif self.settings.dispersion == 'd3bj':
                try:
                    calc = calc.apply(dft.D3Disp(mol, version='d3bj'))
                except:
                    print("Warning: D3BJ dispersion correction not available")
        calc.conv_tol = self.settings.convergence_threshold
        calc.max_cycle = self.settings.max_iterations
        calc.verbose = 0
        return calc 
