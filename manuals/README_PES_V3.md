p   # PES Calculator v3.0: Advanced Conformational Space Exploration

## Overview

This is a completely rewritten potential energy surface (PES) calculator that addresses fundamental issues in the original implementation and provides multiple, chemically-intelligent approaches for conformational space mapping. The new version is suitable for cross-application to all small molecules and provides robust, publication-quality results.

## **🚀 Key Improvements Over Original**

### **Scientific Fixes**
- ✅ **Correct quantum chemistry**: Proper MP2/CCSD implementation building on SCF
- ✅ **Unit consistency**: Hartree throughout, avoiding kcal/mol conversion errors
- ✅ **Dispersion corrections**: Proper D3/D3BJ implementation
- ✅ **Chemical validation**: Geometry checks and convergence monitoring

### **Methodological Advances**
- 🧪 **RDKit integration**: Uses experimental torsion preferences and chemical knowledge
- 🔄 **Multiple strategies**: RDKit ETKDG, systematic scanning, hybrid approaches
- 🎯 **Smart sampling**: Automatic rotatable bond identification and clustering
- 📊 **Quality control**: RMSD-based duplicate removal and energy filtering

### **Usability Improvements**
- 💻 **Universal input**: SMILES, structure files (.mol, .mol2, .sdf) 
- ⚙️ **Flexible configuration**: Comprehensive command-line interface
- 📁 **Rich output**: JSON, DAT, and visualization files
- 🔍 **Detailed logging**: Progress tracking and error reporting

## **📦 Installation**

### Prerequisites
```bash
# Install via conda (recommended)
conda install -c conda-forge rdkit pyscf matplotlib scipy

# Or via pip
pip install rdkit pyscf matplotlib scipy scikit-learn
```

### Verification
```bash
python test_pes_v3_example.py
```

## **🎯 Quick Start**

### Basic Usage
```bash
# Generate conformers of butane using RDKit
python src/1_pes_v3.py "CCCC" --method rdkit --num-conformers 100

# Systematic scanning of ethanol
python src/1_pes_v3.py "CCO" --method systematic --angle-step 30

# High-level calculation from structure file
python src/1_pes_v3.py molecule.mol --quantum-method b3lyp --basis 6-311g-d-p --optimize
```

### Typical Workflow
1. **Generate conformers** with appropriate method
2. **Calculate energies** with desired quantum level
3. **Analyze topology** with existing `2_ph.py` script
4. **Validate results** against chemical intuition/literature

## **🧪 Conformer Generation Methods**

### **RDKit ETKDG (Recommended)**
Uses experimental torsion angle preferences and distance geometry for chemically-intelligent conformer generation.

```bash
python src/1_pes_v3.py "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --method rdkit --num-conformers 200
```

**Best for:** Drug-like molecules, general use, initial surveys

### **Systematic Torsion Scanning**
Exhaustively samples all combinations of rotatable bond angles.

```bash
python src/1_pes_v3.py "CCCC" --method systematic --angle-step 60 --max-conformers 1000
```

**Best for:** Small molecules (≤5 rotatable bonds), validation, detailed analysis

### **Hybrid Approach**
Combines RDKit diversity with systematic refinement around promising regions.

```bash
python src/1_pes_v3.py molecule.sdf --method hybrid --num-conformers 100
```

**Best for:** Important studies requiring both diversity and completeness

## **⚛️ Quantum Chemistry Options**

### Method Selection
- **HF**: Fast screening (`--quantum-method hf`)
- **DFT**: Standard choice (`--quantum-method b3lyp`, `--quantum-method wb97x-d`)
- **MP2**: Post-HF correlation (`--quantum-method mp2`)
- **CCSD**: High accuracy (`--quantum-method ccsd`)

### Basis Sets
- **Minimal**: `sto-3g`, `3-21g` (fast screening)
- **Double-ζ**: `6-31g`, `6-31g-d-p` (standard)
- **Triple-ζ**: `6-311g-d-p`, `aug-cc-pvdz` (high quality)

### Example Configurations
```bash
# Fast screening
--quantum-method hf --basis 3-21g

# Standard DFT
--quantum-method b3lyp --basis 6-31g-d-p --dispersion d3

# High-level calculation  
--quantum-method wb97x-d --basis aug-cc-pvdz --optimize
```

## **📊 Output Files**

### Data Files (`data/` directory)
- **`molecule_pes.dat`**: Simple format for TDA analysis (conformer_id, energy_hartree)
- **`molecule_pes_detailed.json`**: Complete results with geometries and metadata

### Visualizations (`figures/` directory)
- **`molecule_pes_analysis.png`**: Energy distributions and conformer analysis

### TDA Integration
```bash
# Run topological analysis on results
python src/2_ph.py --data-file data/molecule_pes.dat
```

## **🎛️ Command Line Reference**

### Required Arguments
```bash
python src/1_pes_v3.py <molecule>
```
- `<molecule>`: SMILES string, structure file, or molecule name

### Conformer Generation
```bash
--method {rdkit,systematic,hybrid}     # Generation strategy (default: rdkit)
--num-conformers INT                   # Number to generate (default: 100)
--energy-window FLOAT                  # Filter window in kcal/mol (default: 10.0)
```

### Quantum Chemistry
```bash
--quantum-method STR                   # QM method (default: b3lyp)
--basis STR                           # Basis set (default: 6-311g(d,p))
--dispersion {none,d3,d3bj}           # Dispersion correction (default: none)
--charge INT                          # Molecular charge (default: 0)
--multiplicity INT                    # Spin multiplicity (default: 1)
--optimize                            # Optimize geometries
--convergence FLOAT                   # SCF threshold (default: 1e-6)
```

### Systematic Scanning
```bash
--angle-step INT                      # Torsion step size in degrees (default: 60)
--max-conformers INT                  # Maximum for systematic (default: 1000)
```

### Output Control
```bash
--output-prefix STR                   # Custom output name
--no-save                            # Don't save files (testing)
```

## **📋 Best Practices**

### **1. Method Selection**
| Molecule Size | Rotatable Bonds | Recommended Method | Conformers | QM Level |
|---------------|----------------|-------------------|------------|----------|
| Small (≤20 atoms) | ≤3 | systematic | exhaustive | B3LYP/6-311+G(d,p) |
| Medium (20-50 atoms) | 4-6 | rdkit | 200-500 | B3LYP/6-31G* |
| Large (>50 atoms) | >6 | rdkit + screening | 100-200 | HF→DFT hierarchy |

### **2. Quality Control**
- Start with fast methods (HF/3-21G) for validation
- Check energy convergence and geometry reasonableness
- Compare with literature/experimental data when available
- Use multiple methods for important studies

### **3. Performance Optimization**
- Use appropriate conformer counts (more ≠ better beyond diversity limit)
- Apply energy windows to remove high-energy outliers
- Consider multi-level approaches for large systems
- Monitor computational resources

## **🔬 Validation Examples**

### Butane Conformational Analysis
```bash
# Generate butane conformers and analyze energy landscape
python src/1_pes_v3.py "CCCC" --method rdkit --num-conformers 50 --output-prefix butane
python src/2_ph.py --data-file data/butane_pes.dat

# Expected: Anti conformation (~180°) as global minimum
# Should find gauche conformers at ±60° with ~2-3 kcal/mol higher energy
```

### Ethylene Glycol
```bash
# Known for intramolecular hydrogen bonding
python src/1_pes_v3.py "OCCO" --method systematic --angle-step 30 --output-prefix ethylene_glycol

# Expected: Gauche conformer stabilized by H-bonding
```

## **🚨 Troubleshooting**

### Common Issues
| Problem | Solution |
|---------|----------|
| `ImportError: rdkit` | `conda install -c conda-forge rdkit` |
| `ImportError: pyscf` | `pip install pyscf` |
| SCF convergence failure | Try smaller basis or different convergence threshold |
| Too many conformers | Reduce `--num-conformers` or increase `--energy-window` |
| Slow calculations | Use faster method (HF) or smaller basis set |

### Performance Issues
- **Memory**: Large molecules may require >8GB RAM
- **Time**: MP2/CCSD scale steeply with system size
- **Disk**: JSON files can be large for many conformers

## **🔬 Technical Details**

### Differences from Original Implementation

| Aspect | Original (v1) | New (v3) |
|--------|---------------|----------|
| **Input** | Z-matrix templates only | SMILES, structure files |
| **Conformers** | Manual variable definition | Automatic rotatable bond detection |
| **Sampling** | Linear grid scanning | Chemical intelligence (ETKDG) |
| **QM Implementation** | Incorrect MP2/CCSD | Proper SCF foundation |
| **Units** | Mixed Hartree/kcal | Consistent Hartree |
| **Validation** | Hard-coded distances | Chemical geometry checks |
| **Output** | Basic plots | Rich data + visualizations |

### Algorithm Overview
1. **Parse Input**: SMILES → RDKit Mol → 3D coordinates
2. **Generate Conformers**: ETKDG/systematic/hybrid sampling
3. **Filter & Cluster**: Energy window + RMSD clustering
4. **QM Calculations**: Single-point or optimization
5. **Data Management**: JSON + simple DAT formats
6. **Visualization**: Energy landscapes and distributions

## **📚 Further Reading**

- **`CONFORMER_GENERATION_ANALYSIS.md`**: Detailed methodology analysis
- **`test_pes_v3_example.py`**: Usage examples and method comparison
- **RDKit Documentation**: https://rdkit.readthedocs.io/
- **PySCF Documentation**: https://pyscf.org/

## **🤝 Contributing**

The new implementation is designed to be extensible:
- Add new conformer generation methods in `ConformerGenerator`
- Extend quantum chemistry options in `QuantumCalculator`
- Improve analysis and visualization features
- Add machine learning-guided sampling

## **📄 Citation**

When using this code for publications, please cite:
- Original TDA concept and implementation
- RDKit for conformer generation
- PySCF for quantum chemistry calculations
- Specific methods used (DFT functional, basis set, etc.)

---

**✨ The new PES Calculator v3.0 provides a robust, chemically-intelligent foundation for conformational analysis and topological data analysis of molecular systems. ✨** 
