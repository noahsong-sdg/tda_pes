# 1_pes_clean.py - Streamlined PES Calculator

## Overview

This is a cleaned and optimized version of the original `1_pes_old.py` script. The new version focuses on core functionality with improved architecture, better error handling, and a simplified command-line interface.

## Key Improvements

### Code Architecture
- **Object-oriented design**: Main functionality encapsulated in `PESCalculator` class
- **Eliminated code duplication**: Removed redundant functions and consolidated similar operations  
- **Simplified structure**: Reduced from ~1500 lines to ~400 lines while maintaining functionality
- **Better separation of concerns**: Clear distinction between calculation, data handling, and visualization

### Command-Line Interface
- **Streamlined arguments**: Reduced from 15+ arguments to 8 essential ones
- **Intuitive defaults**: Sensible default values that work for most cases
- **Better help system**: Clear examples and usage instructions
- **Improved validation**: Better error messages and input validation

### Computational Methods
- **Modern defaults**: Uses ωB97X-D/6-311++G(d,p) by default (includes dispersion corrections)
- **Method validation**: Warns about suboptimal method choices (e.g., B3LYP without dispersion)
- **Robust error handling**: Graceful handling of SCF convergence failures
- **Simplified method selection**: Cleaner logic for different levels of theory

### Data Management
- **Consistent file formats**: Standardized output formats for both 1D and 2D scans
- **Better metadata**: Enhanced headers with calculation details
- **Force recalculation option**: `--force` flag to override existing data
- **Automatic directory creation**: Creates `data/` and `figures/` directories as needed

### Removed Features
- **3D PES calculations**: Removed due to complexity and limited practical use
- **Animation generation**: Removed complex GIF creation functionality
- **Constrained optimization**: Removed problematic relaxed scan implementation
- **Legacy compatibility**: Removed backwards compatibility with old argument names
- **Visualization options**: Simplified plotting to essential visualizations only

## Usage Examples

### Basic 1D Scan
```bash
# Use default settings (auto-detect first variable, reasonable range)
python src/1_pes_clean.py butane_template.zmat
```

### Custom 1D Scan
```bash
# Specify variable, range, and number of points
python src/1_pes_clean.py butane.zmat \
    --variables dihedral \
    --ranges 0 360 \
    --points 72
```

### 2D Surface Scan
```bash
# 2D scan with auto-detected variables
python src/1_pes_clean.py pentane_multi_template.zmat --mode 2d

# 2D scan with specific variables and ranges  
python src/1_pes_clean.py pentane_multi_template.zmat \
    --mode 2d \
    --variables dihedral1 dihedral2 \
    --ranges 0 360 0 360 \
    --points 25 25
```

### High-Level Calculations
```bash
# MP2 calculation with larger basis set
python src/1_pes_clean.py molecule.zmat \
    --method mp2 \
    --basis aug-cc-pvdz \
    --points 36
```

## Output Files

### 1D Scans
- `data/{molecule}_pes.dat`: Energy data with metadata header
- `figures/{molecule}_pes.png`: 1D energy plot with minimum marked

### 2D Scans  
- `data/{molecule}_2d_pes.npy`: NumPy archive with grids and metadata
- `figures/{molecule}_2d_pes.png`: Multi-panel visualization (3D surface, contour, histogram, 1D slice)

## Computational Methods Available

1. **HF**: Hartree-Fock (fast, low accuracy)
2. **B3LYP**: Hybrid DFT (warning about missing dispersion corrections)
3. **ωB97X-D**: Range-separated hybrid with dispersion (default, recommended)
4. **MP2**: Second-order perturbation theory (high accuracy, slower)
5. **CCSD**: Coupled cluster (very high accuracy, very slow)

## Error Handling

The script includes robust error handling for:
- Missing or invalid input files
- SCF convergence failures  
- Invalid variable names or ranges
- PySCF import errors
- File I/O errors

## Integration with TDA Pipeline

The output `.dat` files are fully compatible with `2_ph.py` for topological data analysis:

```bash
# Generate PES data
python src/1_pes_clean.py butane_template.zmat

# Perform topological analysis
python src/2_ph.py data/butane_pes.dat
```

## Performance

- **Memory efficient**: Uses numpy arrays and proper memory management
- **Progress tracking**: Clear progress indicators for long calculations
- **Fail-fast**: Quick detection and reporting of errors
- **Optimized defaults**: Reasonable balance between speed and accuracy

## Migration from 1_pes_old.py

Most common usage patterns can be migrated easily:

```bash
# Old syntax
python src/1_pes_old.py molecule.zmat --scan-mode 1d --scan-variable dihedral --scan-range 0 360

# New syntax  
python src/1_pes_clean.py molecule.zmat --variables dihedral --ranges 0 360
```

For more complex workflows or the removed features (3D scans, animations), use the original `1_pes_old.py` or `1_pes_constrained.py` scripts.
