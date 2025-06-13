from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class ConformerData:
    """
    Container for conformer information.
    """
    geometry: np.ndarray  # Cartesian coordinates (natoms, 3)
    energy: float  # Energy in Hartree
    conformer_id: int
    method_info: Dict
    converged: bool = True
    torsions: Optional[Dict] = None  # Torsion angles if applicable

@dataclass
class CalculationSettings:
    """
    Container for quantum chemistry calculation settings.
    """
    method: str = 'b3lyp'
    basis: str = '6-311g(d,p)'
    dispersion: str = 'none'  # 'none', 'd3', 'd3bj' 
    charge: int = 0
    multiplicity: int = 1
    convergence_threshold: float = 1e-6
    max_iterations: int = 100 
