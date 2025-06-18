# PH x PES

A project that develops a **persistent homology (PH) pipeline** for analyzing **n-dimensional potential energy landscapes (PEL)** of small (<30 atoms) molecules. The pipeline supports arbitrary-dimensional conformational spaces with automatic coordinate detection and robust topological analysis.

## Installation
PySCF is used for electronic structure calculations. It is only operable on a Linux operating system. If one obtains a potential energy surface via different means, in theory it should be possible to run the persistence module (src/2_ph.py) on the dataset. 

To run this pipeline, use the following commands:

```bash
git clone https://github.com/noahsong-sdg/tda_pes
```
## Instructions
the `src` directory contains the relevant scripts:
- `1_pes.py` generates a potential energy surface for a given z-matrix file.
- `ph_v2.py` performs n-dimensional topological data analysis on PES data using sublevel filtration.

## Key Features
- **N-Dimensional Analysis**: Supports 1D, 2D, 3D, and higher-dimensional potential energy surfaces
- **Auto-Detection**: Automatically detects coordinate types (dihedral angles, bond lengths, etc.) and units
- **Mixed Coordinates**: Handles datasets with different coordinate types (e.g., bond stretch + angle bend)
- **Periodicity Detection**: Automatically handles periodic vs non-periodic coordinates
- **Robust Methods**: Uses cubical complexes for regular grids and Rips complexes for irregular point clouds


## Running 1_pes.py

The PES calculator accepts Z-matrix template files with variable parameters marked as `{variable_name}` and can perform 1D, 2D, or 3D scans of the potential energy surface.

### Basic Usage
```bash
# 1D scan (default) - scans the first variable found in the Z-matrix file
python src/1_pes.py path/to/molecule_template.zmat

# Single point calculation with default values
python src/1_pes.py molecule_template.zmat --single-point
```

### Command Line Options
- `--scan-mode {1d,2d,3d}`: Choose scanning mode (default: 1d)
- `--scan-variables VAR1 [VAR2] [VAR3]`: Specify which variables to scan
- `--scan-ranges MIN1 MAX1 [MIN2 MAX2] [MIN3 MAX3]`: Set scan ranges for each variable
- `--num-points N [N2] [N3]`: Number of points per variable (default: 37 for 1D, 25 for 2D, 15 for 3D)
- `--animation-frames N`: Number of frames for 3D animations (default: 50)
- `--molecule-name NAME`: Custom name for output files
- `--recalculate`: Force recalculation even if output files exist
- `--visualize-min-config`: Visualize the minimum energy configuration
- `--save-step-plots`: Save individual molecular structure plots

### Examples
```bash
# 1D dihedral scan of pentane (0° to 360°, 37 points)
python src/1_pes.py pentane_template.zmat --scan-variables dihedral_angle --scan-ranges 0 360

# 2D surface plot scanning two dihedral angles
python src/1_pes.py pentane_multi_template.zmat --scan-mode 2d --scan-variables dihedral1 dihedral2 --scan-ranges 0 360 0 360 --num-points 25 25

# 3D animated GIF scanning three variables
python src/1_pes.py pentane_multi_template.zmat --scan-mode 3d --scan-variables dihedral1 dihedral2 dihedral3 --animation-frames 60

# Custom molecule with bond length scan
python src/1_pes.py water_template.zmat --scan-variables r_oh --scan-ranges 0.8 1.2 --num-points 50 --molecule-name water_stretch
```

### Output Files
- `data/{molecule}_pes.dat`: Energy data (1D: angle vs energy; 2D/3D: coordinate grids)
- `plots/{molecule}_pes.png`: Energy plot (1D line plot or 2D surface)
- `plots/{molecule}_pes_animation.gif`: Animated visualization (3D mode)
- Individual molecular structure plots (if `--save-step-plots` is used)

## Running ph_v2.py

The enhanced TDA module performs persistent homology analysis on potential energy surfaces of arbitrary dimensionality. It automatically detects coordinate types, handles periodicity, and chooses appropriate computational methods.

### Basic Usage
```bash
# 1D dihedral scan analysis
python src/ph_v2.py data/butane_pes.dat

# 2D conformational analysis
python src/ph_v2.py data/2d_ramachandran.dat --output ramachandran_analysis

# 3D or higher-dimensional analysis
python src/ph_v2.py data/3d_conformational.dat --output 3d_analysis
```

### Supported Data Formats
The module accepts multi-column data files with energy as the last column:
```
# 1D: single coordinate + energy
angle   energy
0.0     0.005
90.0    0.003
...

# 2D: two coordinates + energy  
phi     psi     energy
0.0     0.0     0.010
30.0    30.0    0.008
...

# 3D: three coordinates + energy
bond_length  angle  dihedral  energy
1.54         109.5  180.0     0.000
1.56         111.0  190.0     0.002
...
```

### Auto-Detection Features
- **Coordinate Types**: 
  - Dihedral angles (0-360°) → `angle_N (degrees)`
  - Bond lengths (0.5-5.0 Å) → `bond_length_N (Angstrom)`  
  - Radians (0-2π) → `angle_N_rad (radians)`
  - Other coordinates → `coord_N (unknown)`
- **Periodicity**: Automatically detects periodic coordinates (>300° range)
- **Grid Structure**: Distinguishes regular grids from irregular point clouds

### Analysis Methods
- **1D**: Cubical complex with periodic/non-periodic detection
- **2D**: Regular grid → 2D cubical complex; Irregular → Rips complex
- **N-D**: Grid detection → N-D cubical complex; Irregular → Rips complex

### Output
```bash
# Console output shows:
Loaded 2D PES data: 14 points
Coordinates:
  bond_length_1 (Angstrom): 1.400 to 1.520
  angle_1 (degrees): 100.000 to 140.000
Energy range: 0.000 to 6.903 kcal/mol

Computing persistence for 2D data...
H0 (Connected Components): 2
H1 (Loops/Cycles): 0  
H2+ (Higher-dim features): 286
```

### Chemical Interpretation
- **H0 features**: Energy basins (local minima regions)
- **H1 features**: Transition cycles connecting basins
- **H2+ features**: Higher-dimensional topological features
- **Persistence**: Energetic significance of topological features

### Examples
```bash
# Ammonia inversion (non-periodic)
python src/ph_v2.py data/nh3_pes.dat

# Butane torsion (periodic dihedral)
python src/ph_v2.py data/butane_pes.dat

# 2D Ramachandran plot
python src/ph_v2.py data/ramachandran_2d.dat

# Bond stretch + angle bend coupling
python src/ph_v2.py data/stretch_bend_2d.dat

# 3D conformational landscape
python src/ph_v2.py data/3d_conformation.dat
```
