# CC(=O)OC1=CC=CC=C1C(=O)O

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

# SMILES for aspirin
smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# Generate multiple conformers
AllChem.EmbedMultipleConfs(mol, numConfs=50, randomSeed=42)
AllChem.MMFFOptimizeMoleculeConfs(mol)

# Save lowest energy conformer
writer = Chem.PDBWriter('aspirin_initial.pdb')
writer.write(mol, confId=0)
writer.close()
