# PH x PES

a project that develops a persistent homology (PH) pipeline for analyzing potential energy landscapes (PEL) of small (<30 atoms) molecules.

## Installation
PySCF is used for electronic structure calculations. It is only operable on a Linux operating system. If one obtains a potential energy surface via different means, in theory it should be possible to run the persistence module (src/2_ph.py) on the dataset. 

To run this pipeline, use the following commands:

```bash
git clone https://github.com/noahsong-sdg/tda_pes
```
## Instructions
the `src` directory contains the relevant scripts:
- 1_pes.py generates a potential energy surface for a given z-matrix file.
- 2_ph.py uses the .dat file to compute a sublevel filtration.


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
