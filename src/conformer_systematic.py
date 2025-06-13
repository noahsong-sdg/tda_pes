import numpy as np
from typing import List, Tuple, Union

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Systematic conformer generation will be limited.")
    RDKIT_AVAILABLE = False

class ConformerGeneratorSystematic:
    """
    Generate conformers by systematic torsion scanning.
    """
    def __init__(self, molecule_input: Union[str, 'Chem.Mol']):
        self.mol = self._parse_molecule(molecule_input)
        self.atoms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
    
    def _parse_molecule(self, molecule_input: Union[str, 'Chem.Mol']) -> 'Chem.Mol':
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for conformer generation")
        if isinstance(molecule_input, Chem.Mol):
            return molecule_input
        elif isinstance(molecule_input, str):
            if molecule_input.endswith(('.mol', '.mol2', '.sdf')):
                if molecule_input.endswith('.mol'):
                    mol = Chem.MolFromMolFile(molecule_input)
                elif molecule_input.endswith('.mol2'):
                    mol = Chem.MolFromMol2File(molecule_input)
                elif molecule_input.endswith('.sdf'):
                    supplier = Chem.SDMolSupplier(molecule_input)
                    mol = next(supplier)
                else:
                    raise ValueError(f"Unsupported file format: {molecule_input}")
            else:
                mol = Chem.MolFromSmiles(molecule_input)
            if mol is None:
                raise ValueError(f"Could not parse molecule: {molecule_input}")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        else:
            raise ValueError("molecule_input must be SMILES string, file path, or RDKit Mol")
    
    def generate_conformers(self, angle_step: int = 60, max_conformers: int = 1000, **kwargs) -> List[np.ndarray]:
        """
        Generate conformers by systematic torsion scanning.
        Returns a list of coordinate arrays (natoms, 3).
        """
        print("Generating conformers using systematic torsion scanning...")
        rotatable_bonds = self._identify_rotatable_bonds()
        print(f"Found {len(rotatable_bonds)} rotatable bonds")
        if not rotatable_bonds:
            conf = self.mol.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())])
            return [coords]
        angles = np.arange(0, 360, angle_step)
        conformers = []
        from itertools import product
        angle_combinations = list(product(angles, repeat=len(rotatable_bonds)))
        if len(angle_combinations) > max_conformers:
            import random
            random.seed(42)
            angle_combinations = random.sample(angle_combinations, max_conformers)
        print(f"Generating {len(angle_combinations)} conformers...")
        for i, torsion_angles in enumerate(angle_combinations):
            try:
                mol_copy = Chem.Mol(self.mol)
                conf = mol_copy.GetConformer()
                for (bond_atoms, _), angle in zip(rotatable_bonds, torsion_angles):
                    from rdkit.Chem import rdMolTransforms
                    rdMolTransforms.SetDihedralDeg(conf, *bond_atoms, angle)
                AllChem.MMFFOptimizeMolecule(mol_copy, maxIters=100)
                coords = np.array([conf.GetAtomPosition(j) for j in range(mol_copy.GetNumAtoms())])
                conformers.append(coords)
            except Exception as e:
                print(f"Failed to generate conformer {i}: {e}")
                continue
        unique_conformers = self._remove_duplicate_conformers(conformers)
        print(f"Generated {len(unique_conformers)} unique conformers")
        return unique_conformers
    
    def _identify_rotatable_bonds(self) -> List[Tuple[Tuple[int, int, int, int], str]]:
        rotatable_bonds = []
        for bond in self.mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            if (bond.GetBondType() == Chem.BondType.SINGLE and
                not bond.IsInRing() and
                self.mol.GetAtomWithIdx(atom1_idx).GetDegree() > 1 and
                self.mol.GetAtomWithIdx(atom2_idx).GetDegree() > 1):
                atom1 = self.mol.GetAtomWithIdx(atom1_idx)
                atom2 = self.mol.GetAtomWithIdx(atom2_idx)
                neighbors1 = [n.GetIdx() for n in atom1.GetNeighbors() if n.GetIdx() != atom2_idx]
                neighbors2 = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != atom1_idx]
                if neighbors1 and neighbors2:
                    torsion_atoms = (neighbors1[0], atom1_idx, atom2_idx, neighbors2[0])
                    bond_description = f"{atom1.GetSymbol()}{atom1_idx}-{atom2.GetSymbol()}{atom2_idx}"
                    rotatable_bonds.append((torsion_atoms, bond_description))
        return rotatable_bonds
    
    def _remove_duplicate_conformers(self, conformers: List[np.ndarray], rmsd_threshold: float = 0.5) -> List[np.ndarray]:
        if len(conformers) <= 1:
            return conformers
        unique_conformers = [conformers[0]]
        for i in range(1, len(conformers)):
            is_unique = True
            for unique_conf in unique_conformers:
                rmsd = self._calculate_rmsd(conformers[i], unique_conf)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            if is_unique:
                unique_conformers.append(conformers[i])
        return unique_conformers
    
    def _calculate_rmsd(self, coords1: np.ndarray, coords2: np.ndarray) -> float:
        coords1_centered = coords1 - np.mean(coords1, axis=0)
        coords2_centered = coords2 - np.mean(coords2, axis=0)
        diff = coords1_centered - coords2_centered
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rmsd 
