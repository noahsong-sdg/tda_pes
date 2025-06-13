import numpy as np
from typing import List, Union
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, rdDistGeom, rdForceFieldHelpers


class ConformerGeneratorRDKit:
    """
    Generate conformers using RDKit's ETKDG method.
    """
    def __init__(self, molecule_input: Union[str, 'Chem.Mol']):
        self.mol = self._parse_molecule(molecule_input)
        self.atoms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
    
    def _parse_molecule(self, molecule_input: Union[str, 'Chem.Mol']) -> 'Chem.Mol':
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
    
    def generate_conformers(self, num_conformers: int = 100, energy_window: float = 10.0, **kwargs) -> List[np.ndarray]:
        """
        Generate conformers using RDKit's ETKDG method.
        Returns a list of coordinate arrays (natoms, 3).
        """
        print(f"Generating {num_conformers} conformers using RDKit ETKDG...")
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = kwargs.get('random_seed', 42)
        params.pruneRmsThresh = kwargs.get('rms_threshold', 0.5)
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        conf_ids = AllChem.EmbedMultipleConfs(self.mol, numConfs=num_conformers, params=params)
        if not conf_ids:
            raise RuntimeError("Failed to generate any conformers")
        print(f"Generated {len(conf_ids)} initial conformers")
        results = []
        energies = []
        for conf_id in conf_ids:
            try:
                converged = AllChem.MMFFOptimizeMolecule(self.mol, confId=conf_id)
                if converged == 0:
                    try:
                        props = AllChem.MMFFGetMoleculeProperties(self.mol)
                        ff = AllChem.MMFFGetMoleculeForceField(self.mol, props, confId=conf_id)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            energies.append(energy)
                        else:
                            energy = conf_id * 0.001
                            energies.append(energy)
                    except Exception as e:
                        print(f"MMFF energy calculation failed for conformer {conf_id}: {e}")
                        energy = conf_id * 0.001
                        energies.append(energy)
                    conf = self.mol.GetConformer(conf_id)
                    coords = np.array([conf.GetAtomPosition(i) for i in range(self.mol.GetNumAtoms())])
                    results.append((conf_id, coords, energy))
            except Exception as e:
                print(f"Failed to optimize conformer {conf_id}: {e}")
                continue
        if not results:
            raise RuntimeError("No conformers successfully optimized")
        energies = np.array([r[2] for r in results])
        min_energy = np.min(energies)
        energy_cutoff = min_energy + energy_window
        filtered_conformers = []
        for conf_id, coords, energy in results:
            if energy <= energy_cutoff:
                filtered_conformers.append(coords)
        print(f"Filtered to {len(filtered_conformers)} conformers within {energy_window} kcal/mol energy window")
        unique_conformers = self._remove_duplicate_conformers(filtered_conformers, rmsd_threshold=kwargs.get('rmsd_threshold', 0.5))
        print(f"Final set: {len(unique_conformers)} unique conformers")
        return unique_conformers
    
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
